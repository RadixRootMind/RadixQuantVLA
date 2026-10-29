// PerRowInt8LinearResidual: per-row INT8 quant + INT8 Linear (no bias)
// + residual add, all in one IPluginV3.
//
// Generic LLM building block for post-some-op linears with residual:
//   - o_proj   (input = attn_output, residual = pre-attn x)
//   - down_proj (when silu·mul is computed externally; input = silu(gate)*up,
//                residual = pre-FFN x)
//
// Pipeline:
//   1. dit_int8_per_row_quant_bf16_to_int8(x_bf16) → x_i8, x_scale
//   2. dit_int8_rowwise_gemm_residual_bf16out(x_i8, weight_i8, ..., residual) → y_bf16
//
// Plugin namespace: "gr00t::v1"
// Plugin name:      "PerRowInt8LinearResidual"
//
// Inputs (runtime):
//   0: x        [B, S, K]  BF16  - pre-linear activation
//   1: residual [B, S, N]  BF16  - added to GEMM output
//
// Plugin fields (baked):
//   "weight_i8"            INT8 (N*K)
//   "weight_scale"         FP32 (N,)
//   "static_act_scale_x"   FP32 (B*S,) opt
//   "N"                    INT32 scalar
//   "K"                    INT32 scalar
//
// Outputs:
//   0: y        [B, S, N]  BF16  - INT8 GEMM result + residual
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class PerRowInt8LinearResidualPlugin : public nvinfer1::IPluginV3,
                                         public nvinfer1::IPluginV3OneCore,
                                         public nvinfer1::IPluginV3OneBuild,
                                         public nvinfer1::IPluginV3OneRuntime {
public:
    PerRowInt8LinearResidualPlugin(
        std::string const& name,
        std::vector<int8_t> weightI8,
        std::vector<float>  weightScale,
        int32_t N, int32_t K,
        int32_t rotBlockSize = 0);
    PerRowInt8LinearResidualPlugin(std::string const& name,
                                     nvinfer1::PluginFieldCollection const* fc);
    PerRowInt8LinearResidualPlugin() = delete;
    PerRowInt8LinearResidualPlugin(PerRowInt8LinearResidualPlugin const&) = delete;
    ~PerRowInt8LinearResidualPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    void ensureWeightsOnDevice();

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t> mWeightI8Host;
    std::vector<uint8_t>  mWeightI4Host;  // W4A8: nibble-packed INT4 weights, unpacked to INT8 on device at load
    std::vector<float>  mWeightScaleHost;
    std::vector<float>  mStaticActScaleXHost;

    void* mWeightI8Device{nullptr};
    void* mWeightScaleDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};

    int32_t mN{0};
    int32_t mK{0};
    // Block-diagonal Hadamard applied between the input and the per-row
    // quantizer. 0/1 = disabled. Must be serialized: configurePlugin does NOT
    // re-run on deserialize and the weights are baked folded with W·Hᵀ, so
    // losing this at deserialize would silently produce garbage.
    // SmoothQuant vectors for the butterfly path: the fold the scheme's _s
    // suffix promises. A fixed rotation cannot absorb them the way a dense
    // matrix does, so they ship beside the weights. At most one is set.
    std::vector<float> mScalePreHost;
    std::vector<float> mScaleChHost;
    void* mScalePreDevice{nullptr};
    void* mScaleChDevice{nullptr};
    int32_t mRotBlockSize{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class PerRowInt8LinearResidualPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    PerRowInt8LinearResidualPluginCreator();
    ~PerRowInt8LinearResidualPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t
