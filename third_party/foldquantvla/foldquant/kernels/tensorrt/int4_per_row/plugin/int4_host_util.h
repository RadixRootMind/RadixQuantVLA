// Host-side helpers for the FoldQuant int4 macro plugins.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace gr00t {

// Round-to-nearest-even truncation of FP32 → BF16 (raw 16-bit). R values are
// normal floats; NaN/Inf handling is not required here.
inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

inline std::vector<uint16_t> f32_to_bf16_vec(std::vector<float> const& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = f32_to_bf16(src[i]);
    return out;
}

}  // namespace gr00t
