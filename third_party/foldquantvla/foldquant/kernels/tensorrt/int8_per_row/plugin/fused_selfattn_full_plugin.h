// FusedSelfAttnFull: full self-attention block collapsed into one plugin call:
// AdaLN + per-row quant + INT8 merged QKV GEMM + SDPA + per-row quant +
// INT8 attn_O GEMM + bias + residual.
//
// In the v1 plugin set the same block is split across `FusedAdaLnQuantMergedQkv`
// (emits Q, K, V BF16), a BF16 SDPA subgraph (reshape × 3, transpose × 4,
// MatMul × 2, Mul, Softmax), and a BF16 MatMul attn_O + Add bias + Add residual.
// Here that middle BF16 SDPA chain is replaced by internal cuBLAS batched BF16
// GEMM + custom softmax + cuBLAS batched BF16 GEMM, all inside one plugin call.
//
// Plugin name: "FusedSelfAttnFull", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: x       [B, S, K]   BF16 - hidden state (also reused as residual)
//   1: scale   [B, K]      BF16 - AdaLN scale (per-batch broadcast across S)
//   2: shift   [B, K]      BF16 - AdaLN shift
//
// Plugin fields:
//   "weight_qkv_i8"              INT8 length = 3 * inner_dim * K
//   "weight_qkv_scale"           FP32 length = 3 * inner_dim
//   "bias_qkv"                   FP32 length = 3 * inner_dim
//   "weight_o_i8"                INT8 length = K * inner_dim
//   "weight_o_scale"             FP32 length = K
//   "bias_o"                     FP32 length = K
//   "static_act_scale_x"         FP32 length = max_S  (optional)
//   "static_act_scale_post_sdpa" FP32 length = max_S  (optional)
//   "inner_dim"                  INT32  = num_heads * head_dim
//   "K"                          INT32  = hidden dim
//   "num_heads"                  INT32
//   "head_dim"                   INT32
//   "eps"                        FP32
//
// Outputs:
//   0: out     [B, S, K]   BF16 - post-attn block output (residual added)
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedSelfAttnFullPlugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
public:
    FusedSelfAttnFullPlugin(std::string const& name,
        std::vector<int8_t> wQKV, std::vector<float> wQKVScale, std::vector<float> bQKV,
        std::vector<int8_t> wO,   std::vector<float> wOScale,   std::vector<float> bO,
        int32_t innerDim, int32_t K, int32_t numHeads, int32_t headDim, float eps);
    FusedSelfAttnFullPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedSelfAttnFullPlugin() = delete;
    FusedSelfAttnFullPlugin(FusedSelfAttnFullPlugin const&) = delete;
    ~FusedSelfAttnFullPlugin() override;

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
    void ensureCublasHandle();

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t> mQKVI8Host;
    std::vector<float>  mQKVScaleHost;
    std::vector<float>  mQKVBiasHost;
    std::vector<int8_t> mOI8Host;
    std::vector<float>  mOScaleHost;
    std::vector<float>  mOBiasHost;
    std::vector<float>  mStaticActScaleXHost;
    std::vector<float>  mStaticActScalePostSdpaHost;

    // FoldQuant: block-diagonal butterfly plus the RAW-frame SmoothQuant vectors for
    // the block's two GEMM inputs: the post-adaLN X into the Q/QKV projection,
    // and the post-SDPA context into the output projection. Zero block size is
    // the unrotated path; there is no third state and no silent skip.
    int32_t mRotBlockSize{0};
    std::vector<float>  mScalePreInHost;   // (K,)         post-adaLN channel scale
    std::vector<float>  mScalePreOHost;    // (inner_dim,) post-SDPA channel scale
    void* mScalePreInDevice{nullptr};
    void* mScalePreODevice{nullptr};

    void* mQKVI8Device{nullptr};
    void* mQKVScaleDevice{nullptr};
    void* mQKVBiasDevice{nullptr};
    void* mOI8Device{nullptr};
    void* mOScaleDevice{nullptr};
    void* mOBiasDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};
    void* mStaticActScalePostSdpaDevice{nullptr};

    // cuBLAS handle for batched BF16 GEMM. Lazy-init on first enqueue.
    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mNumHeads{0};
    int32_t mHeadDim{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedSelfAttnFullPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedSelfAttnFullPluginCreator();
    ~FusedSelfAttnFullPluginCreator() override = default;

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
