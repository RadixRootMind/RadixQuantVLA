// FusedAdaLnQuantCrossAttnPrequantized: cross-attn prologue plugin that
// consumes a pre-quantized (int8_enc, scale_enc) pair from an upstream
// `EncoderPreQuant` node instead of re-quantizing the encoder in every block.
// Saves 15 redundant per-row quants per forward (16 cross-attn blocks − 1).
//
// Plugin name: "FusedAdaLnQuantCrossAttnPrequantized", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: x         [B, S, K]          BF16  - hidden states (1536 dim)
//   1: scale     [B, K]             BF16  - AdaLN scale
//   2: shift     [B, K]             BF16  - AdaLN shift
//   3: enc_i8    [B, S_enc, K_enc]  INT8  - pre-quantized encoder
//   4: enc_scale [B, S_enc]         FP32  - per-row scale
//
// Plugin fields:
//   "weight_q_i8"        INT8 length = inner_dim * K
//   "weight_q_scale"     FP32 length = inner_dim
//   "bias_q"             FP32 length = inner_dim
//   "weight_kv_i8"       INT8 length = 2 * inner_dim * K_enc
//   "weight_kv_scale"    FP32 length = 2 * inner_dim
//   "bias_kv"            FP32 length = 2 * inner_dim
//   "static_act_scale_x" FP32 length = max_S (optional, for x quant only)
//   "inner_dim"          INT32
//   "K"                  INT32
//   "K_enc"              INT32
//   "eps"                FP32
//
// Outputs:
//   0: Q       [B, S, inner_dim]      BF16
//   1: K_out   [B, S_enc, inner_dim]  BF16
//   2: V       [B, S_enc, inner_dim]  BF16
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedAdaLnQuantCrossAttnPrequantizedPlugin :
    public nvinfer1::IPluginV3,
    public nvinfer1::IPluginV3OneCore,
    public nvinfer1::IPluginV3OneBuild,
    public nvinfer1::IPluginV3OneRuntime {
public:
    FusedAdaLnQuantCrossAttnPrequantizedPlugin(
        std::string const& name,
        std::vector<int8_t> wQ, std::vector<float> wQScale, std::vector<float> bQ,
        std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
        int32_t innerDim, int32_t K, int32_t KEnc, float eps);
    FusedAdaLnQuantCrossAttnPrequantizedPlugin(std::string const& name,
        nvinfer1::PluginFieldCollection const* fc);
    FusedAdaLnQuantCrossAttnPrequantizedPlugin() = delete;
    FusedAdaLnQuantCrossAttnPrequantizedPlugin(
        FusedAdaLnQuantCrossAttnPrequantizedPlugin const&) = delete;
    ~FusedAdaLnQuantCrossAttnPrequantizedPlugin() override;

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

    std::vector<int8_t> mQI8Host;
    std::vector<float>  mQScaleHost;
    std::vector<float>  mQBiasHost;
    std::vector<int8_t> mKVI8Host;
    std::vector<float>  mKVScaleHost;
    std::vector<float>  mKVBiasHost;
    std::vector<float>  mStaticActScaleXHost;

    void* mQI8Device{nullptr};
    void* mQScaleDevice{nullptr};
    void* mQBiasDevice{nullptr};
    void* mKVI8Device{nullptr};
    void* mKVScaleDevice{nullptr};
    void* mKVBiasDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mKEnc{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedAdaLnQuantCrossAttnPrequantizedPluginCreator :
    public nvinfer1::IPluginCreatorV3One {
public:
    FusedAdaLnQuantCrossAttnPrequantizedPluginCreator();
    ~FusedAdaLnQuantCrossAttnPrequantizedPluginCreator() override = default;

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
