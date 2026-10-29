// FusedCrossAttnFull: full cross-attention block collapsed into one plugin call:
// AdaLN + per-row quant on x + Q GEMM + KV GEMM (from pre-quantized encoder) +
// SDPA (optionally masked) + per-row quant + attn_O GEMM + bias + residual.
//
// In the v1 plugin set the same block is split across
// `FusedAdaLnQuantCrossAttnPrequantized` (emits Q, K, V), a BF16 SDPA subgraph
// (reshape × 3, transpose × 4, MatMul × 2, Mul, Softmax), and a BF16 MatMul
// attn_O + Add bias + Add residual. Here the BF16 SDPA chain is replaced by
// internal cuBLAS batched BF16 GEMM + masked softmax + cuBLAS batched BF16 GEMM.
//
// Plugin name: "FusedCrossAttnFull", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: x          [B, S, K]              BF16 - hidden state (also residual)
//   1: scale      [B, K]                 BF16 - AdaLN scale
//   2: shift      [B, K]                 BF16 - AdaLN shift
//   3: enc_i8     [B, S_enc, K_enc/4]    INT32 - pre-quantized encoder (packed)
//   4: enc_scale  [B, S_enc]             FP32 - per-row encoder scale
//   5: attn_mask  [B, 1, 1, S_enc]       BF16 - additive mask (0=keep, -1e4=mask).
//                                              Required input: pass a zero mask
//                                              tensor for unmasked behavior.
//
// Outputs (depends on `emit_kv` attribute):
//   0: out        [B, S, K]              BF16 - post-cross-attn block output
//   1: kv_bf16    [B, S_enc, 2*K]        BF16 - emitted only when emit_kv != 0;
//                                              K at offset 0..inner_dim, V at
//                                              offset inner_dim..2*inner_dim,
//                                              row-stride = 2*inner_dim.
//                                              Consumed by FusedCrossAttnFullCached
//                                              to skip the KV GEMM on subsequent
//                                              denoising steps (KV-cache reuse).
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedCrossAttnFullPlugin : public nvinfer1::IPluginV3,
                                 public nvinfer1::IPluginV3OneCore,
                                 public nvinfer1::IPluginV3OneBuild,
                                 public nvinfer1::IPluginV3OneRuntime {
public:
    FusedCrossAttnFullPlugin(std::string const& name,
        std::vector<int8_t> wQ,  std::vector<float> wQScale,  std::vector<float> bQ,
        std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
        std::vector<int8_t> wO,  std::vector<float> wOScale,  std::vector<float> bO,
        int32_t innerDim, int32_t K, int32_t KEnc,
        int32_t numHeads, int32_t headDim, float eps);
    FusedCrossAttnFullPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedCrossAttnFullPlugin() = delete;
    FusedCrossAttnFullPlugin(FusedCrossAttnFullPlugin const&) = delete;
    ~FusedCrossAttnFullPlugin() override;

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

    std::vector<int8_t> mQI8Host, mKVI8Host, mOI8Host;
    std::vector<float>  mQScaleHost, mKVScaleHost, mOScaleHost;
    std::vector<float>  mQBiasHost,  mKVBiasHost,  mOBiasHost;
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

    void* mQI8Device{nullptr};   void* mKVI8Device{nullptr};  void* mOI8Device{nullptr};
    void* mQScaleDevice{nullptr}; void* mKVScaleDevice{nullptr}; void* mOScaleDevice{nullptr};
    void* mQBiasDevice{nullptr};  void* mKVBiasDevice{nullptr};  void* mOBiasDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};
    void* mStaticActScalePostSdpaDevice{nullptr};

    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mKEnc{0};
    int32_t mNumHeads{0};
    int32_t mHeadDim{0};
    float   mEps{1e-5f};
    bool    mHasMask{false};
    int32_t mHasMaskSerialized{0};   // int32 mirror of mHasMask; passed by pointer to PluginField.
    int32_t mEmitKV{0};              // 0 = legacy 1-output; 1 = emit KV cache as 2nd output.

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedCrossAttnFullPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedCrossAttnFullPluginCreator();
    ~FusedCrossAttnFullPluginCreator() override = default;

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
