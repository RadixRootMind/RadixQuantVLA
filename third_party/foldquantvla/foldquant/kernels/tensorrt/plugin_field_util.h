// Plugin-field length normalization shared by every FoldQuant TensorRT plugin.
#pragma once

#include <NvInferRuntime.h>

#include <cstddef>

namespace gr00t {

// Byte attributes reach a plugin ctor through two different length
// conventions. The ONNX parser's fallback plugin importer hands a STRING
// attribute over as a kCHAR field whose length is the payload size in BYTES
// (plus the string's NUL terminator on some parser builds); deserialization
// hands over the typed field serializeToFields() wrote, whose length is the
// ELEMENT count. Reading f.length as elements on the parser path over-reads
// the payload sizeof(elem)x. That is harmless where only the prefix is consumed, but
// it bloats every serialized engine by the same factor and breaks any size
// validation (PerRowInt4LinearResidual's dense-rotation check rejected every
// correctly-baked expert engine). Normalize to an element count.
//
// A 4-byte payload disambiguates the NUL by modulo; 1-byte payloads cannot and
// keep their raw byte count (at most one trailing byte, never consumed).
inline size_t fieldElemCount(nvinfer1::PluginField const& f, size_t elemSize) {
    if (f.type == nvinfer1::PluginFieldType::kCHAR) {
        size_t bytes = static_cast<size_t>(f.length);
        if (elemSize > 1 && bytes % elemSize == 1) bytes -= 1;
        return bytes / elemSize;
    }
    return static_cast<size_t>(f.length);
}

}  // namespace gr00t
