// FusedCrossAttnFullCached: cached-KV variant of FusedCrossAttnFull. Skips the
// per-step KV GEMM by consuming pre-computed K, V tensors emitted by
// FusedCrossAttnFull at denoising step 0 (with `emit_kv = 1`). Used at
// denoising steps 1..N-1 of the unfolded KV-cache loop, where the encoder
// features don't change so K, V can be reused across steps.
//
// Plugin name: "FusedCrossAttnFullCached", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: x          [B, S, K]              BF16 - hidden state (also residual)
//   1: scale      [B, K]                 BF16 - AdaLN scale
//   2: shift      [B, K]                 BF16 - AdaLN shift
//   3: kv_bf16    [B, S_enc, 2*K]        BF16 - cached K,V interleaved
//                                              (K at offset 0..inner_dim,
//                                              V at offset inner_dim..2*inner_dim,
//                                              row-stride = 2*inner_dim).
//                                              Same layout as FusedCrossAttnFull
//                                              output 1.
//   4: attn_mask  [B, 1, 1, S_enc]       BF16 - additive mask (optional; pass
//                                              4-input form to skip).
//
// Outputs:
//   0: out        [B, S, K]              BF16 - post-cross-attn block output
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedCrossAttnFullCachedPlugin : public nvinfer1::IPluginV3,
                                       public nvinfer1::IPluginV3OneCore,
                                       public nvinfer1::IPluginV3OneBuild,
                                       public nvinfer1::IPluginV3OneRuntime {
public:
    FusedCrossAttnFullCachedPlugin(std::string const& name,
        std::vector<int8_t> wQ, std::vector<float> wQScale, std::vector<float> bQ,
        std::vector<int8_t> wO, std::vector<float> wOScale, std::vector<float> bO,
        int32_t innerDim, int32_t K,
        int32_t numHeads, int32_t headDim, float eps);
    FusedCrossAttnFullCachedPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedCrossAttnFullCachedPlugin() = delete;
    FusedCrossAttnFullCachedPlugin(FusedCrossAttnFullCachedPlugin const&) = delete;
    ~FusedCrossAttnFullCachedPlugin() override;

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

    // No K/V weights; KV is consumed from inputs[3].
    std::vector<int8_t> mQI8Host, mOI8Host;
    std::vector<float>  mQScaleHost, mOScaleHost;
    std::vector<float>  mQBiasHost,  mOBiasHost;
    std::vector<float>  mStaticActScaleXHost;
    std::vector<float>  mStaticActScalePostSdpaHost;

    void* mQI8Device{nullptr};  void* mOI8Device{nullptr};
    void* mQScaleDevice{nullptr}; void* mOScaleDevice{nullptr};
    void* mQBiasDevice{nullptr};  void* mOBiasDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};
    void* mStaticActScalePostSdpaDevice{nullptr};

    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mNumHeads{0};
    int32_t mHeadDim{0};
    float   mEps{1e-5f};
    bool    mHasMask{false};
    int32_t mHasMaskSerialized{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedCrossAttnFullCachedPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedCrossAttnFullCachedPluginCreator();
    ~FusedCrossAttnFullCachedPluginCreator() override = default;

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
