"""Compare the native library to eager upstream inference using synthetic weights."""
import argparse
import ctypes as C
import sys
import os
from pathlib import Path

import torch
from build_architecture import make_generator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("engine", choices=["applio", "rvc"])
parser.add_argument("source", type=Path)
parser.add_argument("architecture", type=Path)
parser.add_argument("library", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
meta = {"rate": "", "frames": "", "engine": ""}
torch.jit.load(str(args.architecture), _extra_files=meta)
rate, n = int(meta["rate"]), int(meta["frames"])
torch.set_num_threads(2)
torch.manual_seed(810)
model = make_generator(args.engine, rate)
lib = C.CDLL(str(args.library.resolve()))
ptr = C.c_void_p

def api(name, result, *params):
    f = getattr(lib, name)
    f.restype = result
    f.argtypes = params
    return f

error = api("voice_error", C.c_char_p)
create = api("voice_open", ptr, C.c_char_p, C.c_int)
close = api("voice_close", None, ptr)
weight = api("voice_set_weight", C.c_int, ptr, C.c_char_p, ptr, ptr, C.c_size_t, C.c_size_t)
finish = api("voice_finish", C.c_int, ptr)
infer = api("voice_infer", C.c_int, ptr, ptr, ptr, ptr, C.c_int64, C.c_int64, ptr, C.c_int64)
handle = create(os.fsencode(args.architecture), 0)
assert handle, error()
try:
    assert finish(handle) == -1
    for k, t in model.generator.named_parameters():
        shape = (C.c_int64 * t.ndim)(*t.shape)
        rc = weight(handle, k.encode(), t.contiguous().data_ptr(), shape, t.ndim, t.numel())
        assert rc == 0, error()
    assert finish(handle) == 0, error()
    x = torch.randn(1, n, 768)
    coarse = torch.full((1,n), 92)
    for freq in (0., 192.):
        hz = torch.full((1,n), freq)
        y = torch.empty(1, 1, n * rate // 100)
        torch.manual_seed(160)
        with torch.no_grad():
            target = model(x, torch.tensor([n]), coarse, hz, torch.tensor([0]))
        torch.manual_seed(160)
        rc = infer(handle,x.data_ptr(),coarse.data_ptr(),hz.data_ptr(),n,0,y.data_ptr(),y.numel())
        assert rc == 0, error()
        torch.testing.assert_close(y,target,atol=2e-5,rtol=2e-4)
        print(args.engine, rate, freq, "max absolute error", (target-y).abs().max().item())
finally:
    close(handle)
