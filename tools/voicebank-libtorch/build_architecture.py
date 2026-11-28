"""Build a fixed-window, untrained CPU architecture for native LibTorch inference."""
import argparse
import sys
from pathlib import Path

import torch


class VoiceGenerator(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, phone, lengths, coarse, hz, speaker):
        return self.generator.infer(phone, lengths, coarse, hz, speaker)[0]


def make_generator(engine, rate):
    rates, kernels = ([12, 10, 2, 2], [24, 20, 4, 4]) if rate == 48000 else ([10, 10, 2, 2], [16, 16, 4, 4])
    config = (1025, 32, 192, 192, 768, 2, 6, 3, 0., "1", [3, 7, 11],
              [[1, 3, 5]] * 3, rates, 512, kernels, 109, 256, rate)
    if engine == "applio":
        from rvc.lib.algorithm.synthesizers import Synthesizer
        # The public infer method is invoked by the traced wrapper. Disable its
        # additional scripting entry point during this isolated build process.
        if hasattr(Synthesizer.infer, "_torchscript_modifier"):
            delattr(Synthesizer.infer, "_torchscript_modifier")
        model = Synthesizer(*config, use_f0=True, text_enc_hidden_dim=768)
    else:
        from infer.module.models import SynthesizerTrnMs768NSFsid
        model = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    del model.enc_q
    return VoiceGenerator(model).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("applio", "rvc"))
    parser.add_argument("source", type=Path)
    parser.add_argument("rate", type=int, choices=(40000, 48000))
    parser.add_argument("frames", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(734)
    model = make_generator(args.engine, args.rate)
    n = args.frames
    x = (torch.randn(1, n, 768), torch.tensor([n]), torch.full((1, n), 100),
         torch.full((1, n), 180.), torch.tensor([0]))
    with torch.no_grad():
        traced = torch.jit.trace(model, x, check_trace=False)
        if set(dict(traced.named_parameters())) != set(dict(model.named_parameters())):
            raise RuntimeError("tracing changed parameter names")
        # Compare the reloaded program to upstream inference on different inputs.
        # Save the compiled operators only with placeholder weights. The runtime
        # must replace every parameter before the first live inference.
        program = traced.save_to_buffer()
        import io
        restored = torch.jit.load(io.BytesIO(program))
        for voice_hz in (0., 233.):
            test = (torch.randn_like(x[0]), x[1], x[2], torch.full_like(x[3], voice_hz), x[4])
            torch.manual_seed(42)
            expected = model(*test)
            torch.manual_seed(42)
            actual = restored(*test)
            if expected.numel() != n * args.rate // 100:
                raise RuntimeError(f"unexpected output shape {expected.shape}")
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        for p in traced.parameters():
            p.zero_()
        traced.save(str(args.output), _extra_files={
            "engine": args.engine, "rate": str(args.rate), "frames": str(n)
        })


if __name__ == "__main__":
    main()
