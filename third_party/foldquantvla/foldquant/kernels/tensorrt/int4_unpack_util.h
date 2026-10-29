#pragma once
// Host-side INT4 nibble unpack shared by the INT8 per-row plugins' W4A8 mode.
//
// The packing is rotation.pack_int4_nibbles (the single producer): for
// each output row, byte b holds column 2b in the LOW nibble and column 2b+1 in
// the HIGH nibble, two's-complement signed. The unpacked INT8 tensor keeps the
// INT4 grid (-8..7) and the per-row INT4 scale, so an INT8 GEMM over it is the
// exact W4A8 arithmetic: INT4 weights, INT8 per-token activations.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gr00t {

inline std::vector<int8_t> unpackInt4Nibbles(std::vector<uint8_t> const& packed) {
    std::vector<int8_t> out(packed.size() * 2);
    for (size_t b = 0; b < packed.size(); ++b) {
        uint8_t const byte = packed[b];
        out[2 * b]     = static_cast<int8_t>(static_cast<int8_t>(byte << 4) >> 4);  // low nibble, sign-extended
        out[2 * b + 1] = static_cast<int8_t>(static_cast<int8_t>(byte & 0xF0) >> 4); // high nibble, sign-extended
    }
    return out;
}

}  // namespace gr00t
