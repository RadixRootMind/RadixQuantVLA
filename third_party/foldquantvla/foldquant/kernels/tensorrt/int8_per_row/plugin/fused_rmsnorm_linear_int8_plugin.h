// FusedRmsNormLinearInt8: RMSNorm + per-row INT8 quant + INT8 Linear (no bias).
//
// Generic LLM building block. Replaces:
//   y_n = RMSNorm(x, gamma, eps)         // gamma is per-channel affine
//   y   = y_n @ W.T                       // no bias (Qwen3 convention)
//
// Suitable for any Qwen3 Linear layer that follows a RMSNorm:
//   - q_proj / k_proj / v_proj (after input_layernorm)
//   - gate_proj / up_proj      (after post_attention_layernorm)
//
// For merged GEMM (Q+K+V or gate+up), set N = sum of output dims and bake the
// concatenated [W_Q ; W_K ; V] weight into weight_i8. The caller then splits
// the output BF16 tensor.
//
// Plugin namespace: "gr00t::v1"
// Plugin name:      "FusedRmsNormLinearInt8"
//
// Inputs (runtime):
//   0: x      [B, S, K]  BF16  - hidden_states
//
// Plugin fields (baked at engine build):
//   "weight_i8"            INT8 (N*K)        - weight transposed, row-major (N, K)
//   "weight_scale"         FP32 (N,)          - per-output-channel scale
//   "gamma"                BF16 (K,)          - RMSNorm.weight (per-channel)
//   "static_act_scale_x"   FP32 (B*S,) opt    - empty = dynamic per-row amax
//   "N"                    INT32 scalar       - output dim
//   "K"                    INT32 scalar       - input  dim (= hidden_size)
//   "eps"                  FP32 scalar        - RMSNorm epsilon (Qwen3: 1e-6)
//
// Outputs:
//   0: y      [B, S, N]  BF16
//
// Underlying kernels:
//   - rmsnorm_per_row_quant_bf16_to_int8(_static_)  (in rmsnorm_per_row_quant.h)
//   - dit_int8_rowwise_gemm_bf16out                 (in dit_int8_rowwise_v2.h)
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedRmsNormLinearInt8Plugin : public nvinfer1::IPluginV3,
                                      public nvinfer1::IPluginV3OneCore,
                                      public nvinfer1::IPluginV3OneBuild,
                                      public nvinfer1::IPluginV3OneRuntime {
public:
    FusedRmsNormLinearInt8Plugin(
        std::string const& name,
        std::vector<int8_t> weightI8,
        std::vector<float>  weightScale,
        std::vector<uint16_t> gammaBf16,
        int32_t N,
        int32_t K,
        float eps,
        int32_t rotBlockSize = 0);
    FusedRmsNormLinearInt8Plugin(std::string const& name,
                                  nvinfer1::PluginFieldCollection const* fc);
    FusedRmsNormLinearInt8Plugin() = delete;
    FusedRmsNormLinearInt8Plugin(FusedRmsNormLinearInt8Plugin const&) = delete;
    ~FusedRmsNormLinearInt8Plugin() override;

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

    std::vector<int8_t>   mWeightI8Host;
    std::vector<uint8_t>  mWeightI4Host;  // W4A8: nibble-packed INT4 weights, unpacked to INT8 on device at load
    std::vector<float>    mWeightScaleHost;
    std::vector<uint16_t> mGammaBf16Host;          // raw BF16 storage (uint16_t)
    std::vector<float>    mStaticActScaleXHost;     // optional, empty = dynamic

    void* mWeightI8Device{nullptr};
    void* mWeightScaleDevice{nullptr};
    void* mGammaDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};

    int32_t mN{0};
    int32_t mK{0};
    float   mEps{1e-6f};
    // Block-diagonal Hadamard applied between RMSNorm and the per-row quantizer.
    // 0/1 = disabled. Must be serialized: configurePlugin does NOT re-run on
    // deserialize, and the weights are baked already folded with W·Hᵀ, so a
    // rot_block_size lost at deserialize would silently produce garbage.
    int32_t mRotBlockSize{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedRmsNormLinearInt8PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedRmsNormLinearInt8PluginCreator();
    ~FusedRmsNormLinearInt8PluginCreator() override = default;

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
