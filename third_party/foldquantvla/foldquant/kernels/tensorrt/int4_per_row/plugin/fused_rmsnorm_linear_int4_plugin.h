// FusedRmsNormLinearInt4: RMSNorm + FWHT + per-row INT4 quant + INT4 Linear.
//
// INT4 sibling of FusedRmsNormLinearInt8, for the LLM W4A4 path
// (`w4a4_srg`). Replaces:
//   y_n = RMSNorm(x, gamma, eps)          // gamma is per-channel affine
//   y   = y_n @ W.T                        // no bias (Qwen3 convention)
//
// Used at the two merged-GEMM sites of every Qwen3 layer:
//   - q_proj + k_proj + v_proj (after input_layernorm),          N = 4096
//   - gate_proj + up_proj      (after post_attention_layernorm),  N = 12288
// Bake the concatenated weight into weight_i4; the caller splits the BF16 output.
//
// Differences from the INT8 plugin, and only these:
//   - weight is packed INT4 (N*K/2 bytes) + per-output-channel FP32 scale;
//   - the activation quantizer is INT4 (symmetric [-7, 7], never -8);
//   - there is NO static_act_scale_x. W4A4 is dynamic-only: static per-tensor
//     activation scale is what collapsed `int8_w8a8` to SR 0.561, and the INT8
//     path already guard-rejected static x rotation as unsound.
//
// Plugin namespace: "gr00t::v1"
// Plugin name:      "FusedRmsNormLinearInt4"
//
// Inputs (runtime):
//   0: x      [B, S, K]  BF16  - hidden_states
//
// Plugin fields (baked at engine build):
//   "weight_i4"        INT8 (N*K/2)   - packed INT4, row-major (N, K),
//                                       element i -> byte i/2, even i = low nibble
//   "weight_scale"     FP32 (N,)      - per-output-channel scale
//   "gamma"            BF16 (K,)      - RMSNorm.weight (per-channel)
//   "N"                INT32 scalar   - output dim
//   "K"                INT32 scalar   - input  dim (= hidden_size)
//   "eps"              FP32 scalar    - RMSNorm epsilon (Qwen3: 1e-6)
//   "rot_block_size"   INT32 scalar   - block-Hadamard size (0/1 = disabled)
//
// Outputs:
//   0: y      [B, S, N]  BF16
//
// Underlying kernels (both in dit_int4_rowwise.h):
//   - rmsnorm_fwht_per_row_quant_bf16_to_int4
//   - dit_int4_rowwise_gemm_bias_bf16out, fed a zero bias; the no-bias entry
//     point is declared in that header but has no implementation. See
//     mZeroBiasDevice below.
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedRmsNormLinearInt4Plugin : public nvinfer1::IPluginV3,
                                      public nvinfer1::IPluginV3OneCore,
                                      public nvinfer1::IPluginV3OneBuild,
                                      public nvinfer1::IPluginV3OneRuntime {
public:
    FusedRmsNormLinearInt4Plugin(
        std::string const& name,
        std::vector<int8_t> weightI4,
        std::vector<float>  weightScale,
        std::vector<uint16_t> gammaBf16,
        int32_t N,
        int32_t K,
        float eps,
        int32_t rotBlockSize = 0,
        float actClipRatio = 1.0f);
    FusedRmsNormLinearInt4Plugin(std::string const& name,
                                  nvinfer1::PluginFieldCollection const* fc);
    FusedRmsNormLinearInt4Plugin() = delete;
    FusedRmsNormLinearInt4Plugin(FusedRmsNormLinearInt4Plugin const&) = delete;
    ~FusedRmsNormLinearInt4Plugin() override;

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
    int ensureWeightsOnDevice();  // cudaError_t as int; 0 = success

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t>   mWeightI4Host;
    std::vector<float>    mWeightScaleHost;
    std::vector<uint16_t> mGammaBf16Host;          // raw BF16 storage (uint16_t)

    void* mWeightI4Device{nullptr};
    void* mWeightScaleDevice{nullptr};
    void* mGammaDevice{nullptr};
    // Qwen3 q/k/v/gate/up have no bias, but the int4 GEMM family only ships the
    // bias and bias+residual EVT epilogues; `dit_int4_rowwise_gemm_bf16out` is
    // declared in dit_int4_rowwise.h and never defined. Rather than compile a
    // third CUTLASS instantiation for a case that is one fp32 add of zero, this
    // holds an N-vector of zeros. Exact, and N floats is at most 48 KB.
    void* mZeroBiasDevice{nullptr};

    int32_t mN{0};
    int32_t mK{0};
    float   mEps{1e-6f};
    // Block-diagonal Hadamard applied between RMSNorm and the per-row quantizer.
    // 0/1 = disabled. Must be serialized: configurePlugin does NOT re-run on
    // deserialize, and the weights are baked already folded with W·Hᵀ, so a
    // rot_block_size lost at deserialize would silently produce garbage.
    int32_t mRotBlockSize{0};
    float mActClipRatio{1.0f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedRmsNormLinearInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedRmsNormLinearInt4PluginCreator();
    ~FusedRmsNormLinearInt4PluginCreator() override = default;

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
