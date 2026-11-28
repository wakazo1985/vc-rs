// Worker-thread LibTorch generator. The caller supplies tensors decoded from the
// original RVC state_dict. The C boundary never transfers exception ownership.
#include <torch/script.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define VOICE_API extern "C" __declspec(dllexport)
#else
#define VOICE_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;

struct Generator {
    torch::jit::Module module;
    std::unordered_map<std::string, torch::Tensor> pending;
    int64_t frames, rate;
    torch::Device device;

    Generator(const char* path, bool cuda) : device(cuda ? torch::kCUDA : torch::kCPU) {
        torch::jit::ExtraFilesMap metadata{{"frames", ""}, {"rate", ""}};
        module = torch::jit::load(path, torch::kCPU, metadata);
        frames = std::stoll(metadata.at("frames"));
        rate = std::stoll(metadata.at("rate"));
        TORCH_CHECK(frames > 0 && (rate == 40000 || rate == 48000), "invalid architecture");
        for (const auto& parameter : module.named_parameters()) {
            const std::string prefix = "generator.";
            TORCH_CHECK(parameter.name.compare(0, prefix.size(), prefix) == 0, "bad parameter name");
            pending.emplace(parameter.name.substr(prefix.size()), parameter.value);
        }
        TORCH_CHECK(!pending.empty(), "empty architecture");
    }
    bool ready = false;
};

template <class F> int checked(F&& operation) {
    try {
        c10::InferenceMode guard;
        operation();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
}
VOICE_API const char* voice_error() { return last_error.c_str(); }
VOICE_API void* voice_open(const char* architecture, int cuda) {
    Generator* handle = nullptr;
    checked([&] { handle = new Generator(architecture, cuda != 0); });
    return handle;
}
VOICE_API void voice_close(Generator* g) { delete g; }

VOICE_API int voice_set_weight(Generator* g, const char* name, const float* data,
                               const int64_t* shape, size_t rank, size_t count) {
    return checked([&] {
        TORCH_CHECK(g && !g->ready && name && data && shape, "invalid weight input");
        std::string key(name);
        auto match = g->pending.find(key);
        if (match == g->pending.end()) {
            const std::pair<std::string, std::string> aliases[] = {
                {".weight_g", ".parametrizations.weight.original0"},
                {".weight_v", ".parametrizations.weight.original1"},
                {".parametrizations.weight.original0", ".weight_g"},
                {".parametrizations.weight.original1", ".weight_v"}
            };
            for (const auto& [a,b] : aliases) {
                if (key.size() >= a.size() && key.compare(key.size()-a.size(), a.size(), a) == 0) {
                    match = g->pending.find(key.substr(0,key.size()-a.size()) + b);
                    break;
                }
            }
        }
        TORCH_CHECK(match != g->pending.end(), "unknown or duplicate weight: ", key);
        auto& target = match->second;
        TORCH_CHECK(target.sizes().equals(at::IntArrayRef(shape, rank)) &&
                    static_cast<size_t>(target.numel()) == count, "invalid weight dimensions: ", key);
        target.copy_(torch::from_blob(const_cast<float*>(data), target.sizes(), torch::kFloat32));
        g->pending.erase(match);
    });
}
VOICE_API int voice_finish(Generator* g) {
    return checked([&] {
        TORCH_CHECK(g && !g->ready, "invalid generator state");
        TORCH_CHECK(g->pending.empty(), "missing ", g->pending.size(), " trained weights");
        g->module.to(g->device);
        g->module.eval();
        g->ready = true;
    });
}
VOICE_API int voice_infer(Generator* g, float* features, int64_t* coarse, float* hz,
                          int64_t frames, int64_t speaker, float* output, int64_t samples) {
    return checked([&] {
        TORCH_CHECK(g && g->ready && features && coarse && hz && output, "invalid inference state");
        TORCH_CHECK(frames == g->frames && speaker >= 0 && speaker < 109 &&
                    samples == frames*(g->rate/100), "input/output shape mismatch");
        auto on_device = [&](auto t) { return t.to(g->device); };
        auto value = g->module.forward({
            on_device(torch::from_blob(features, {1,frames,768}, torch::kFloat32)),
            on_device(torch::tensor({frames},torch::kInt64)),
            on_device(torch::from_blob(coarse,{1,frames},torch::kInt64)),
            on_device(torch::from_blob(hz,{1,frames},torch::kFloat32)),
            on_device(torch::tensor({speaker},torch::kInt64))
        }).toTensor().to(torch::kCPU).contiguous();
        TORCH_CHECK(value.scalar_type() == torch::kFloat32 && value.numel() == samples &&
                    torch::isfinite(value).all().item<bool>(), "invalid synthesis output");
        std::memcpy(output, value.data_ptr<float>(), samples*sizeof(float));
    });
}
