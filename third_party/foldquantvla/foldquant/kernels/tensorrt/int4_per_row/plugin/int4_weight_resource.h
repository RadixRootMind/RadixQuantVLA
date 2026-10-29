// Shared immutable device weights for the FoldQuant int4 macro plugins.
//
// Every macro plugin used to keep a full host copy of its packed weights /
// scales / biases / perms / rotations and lazy-upload a *private* device copy on
// its first enqueue(). attachToContext() ignored IPluginResourceContext and only
// returned clone(), so GPU (and host) memory grew ~linearly with the number of
// TensorRT execution contexts, and every fresh context paid an alloc+upload on
// its first run. That is fine single-context but expensive when serving several
// concurrent contexts against the large W4A4 DiT weights.
//
// SharedDeviceWeights is an IPluginResource: TensorRT refcounts one instance per
// (content-derived) key, so N execution contexts that reference identical weights
// share a single device copy. The immutable weights live here; per-context
// *mutable* state (cuBLAS handle, workspace) stays on the plugin object.
#pragma once

#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

// One immutable host buffer to be uploaded to device and shared across contexts.
struct WeightSpec {
    void const* host;
    size_t bytes;
};

// IPluginResource owning a set of immutable device buffers, uploaded once and
// shared across every execution context / plugin clone that acquires the same key.
class SharedDeviceWeights : public nvinfer1::IPluginResource {
public:
    SharedDeviceWeights() = default;
    // Template ctor (pre-registration): carries the host specs so clone() (which
    // TensorRT calls once per new key) can allocate and upload the device copies.
    explicit SharedDeviceWeights(std::vector<WeightSpec> specs) : mSpecs(std::move(specs)) {}

    ~SharedDeviceWeights() noexcept override { freeAll(); }

    // Note: getInterfaceInfo() is intentionally NOT overridden; the base
    // IPluginResource provides the required {"IPluginResource", 1, 0}.

    // Called by TensorRT exactly once per unique key; only the clone is registered.
    nvinfer1::IPluginResource* clone() noexcept override {
        try {
            auto* r = new SharedDeviceWeights();
            r->mDevice.assign(mSpecs.size(), nullptr);
            for (size_t i = 0; i < mSpecs.size(); ++i) {
                size_t const n = mSpecs[i].bytes;
                if (n == 0) continue;
                if (cudaMalloc(&r->mDevice[i], n) != cudaSuccess) { delete r; return nullptr; }
                if (cudaMemcpy(r->mDevice[i], mSpecs[i].host, n, cudaMemcpyHostToDevice) != cudaSuccess) {
                    delete r;
                    return nullptr;
                }
            }
            return r;
        } catch (...) {
            return nullptr;
        }
    }

    // Called by TensorRT when the last reference to this key is released.
    int32_t release() noexcept override {
        freeAll();
        return 0;
    }

    // Device pointer for the i-th spec (nullptr if empty / out of range).
    void* buf(size_t i) const noexcept { return (i < mDevice.size()) ? mDevice[i] : nullptr; }

private:
    void freeAll() noexcept {
        for (void* p : mDevice) {
            if (p) cudaFree(p);
        }
        mDevice.clear();
    }

    std::vector<WeightSpec> mSpecs;  // valid only on the template (pre-clone)
    std::vector<void*> mDevice;      // owned device buffers (on the registered clone)
};

// FNV-1a 64-bit digest over a plugin tag + every host buffer → a stable key. The
// tag prefix keys resources per plugin type (and keeps the key human-readable);
// the content hash means two layers with different weights never collide, while
// N contexts over the *same* layer resolve to one shared device copy.
inline std::string weightDigest(char const* tag, std::vector<WeightSpec> const& specs) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](void const* p, size_t n) {
        auto const* b = static_cast<uint8_t const*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    };
    mix(tag, std::strlen(tag));
    for (auto const& s : specs) {
        uint64_t const nb = s.bytes;  // length-delimit so buffers can't run together
        mix(&nb, sizeof(nb));
        if (s.host && s.bytes) mix(s.host, s.bytes);
    }
    char out[128];
    std::snprintf(out, sizeof(out), "%s.%016llx", tag, static_cast<unsigned long long>(h));
    return std::string(out);
}

// Acquire (creating on first use) the shared resource for `key` built from `specs`.
// Returns a borrowed pointer owned by the plugin registry; pair every successful
// call with releaseSharedWeights(key) when the plugin instance is destroyed.
inline SharedDeviceWeights* acquireSharedWeights(std::string const& key, std::vector<WeightSpec> specs) {
    auto* reg = ::getPluginRegistry();
    if (reg == nullptr) return nullptr;
    SharedDeviceWeights templ(std::move(specs));
    return static_cast<SharedDeviceWeights*>(reg->acquirePluginResource(key.c_str(), &templ));
}

inline void releaseSharedWeights(std::string const& key) {
    auto* reg = ::getPluginRegistry();
    if (reg != nullptr) reg->releasePluginResource(key.c_str());
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t
