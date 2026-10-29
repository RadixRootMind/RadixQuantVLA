// EncoderPreQuant: runs the per-row INT8 quant of the cross-attention encoder
// once at the start of the graph.
//
// In the single-pass DiT the encoder (BF16, shape (B, S_enc, K_enc)) is identical
// for all 16 cross-attn blocks. The v1 prologue plugin redundantly quantized
// the encoder inside every block; this plugin emits (int8_enc, scale_enc) once
// and the 16 downstream FusedAdaLnQuantCrossAttnPrequantized / FusedCrossAttnFull
// plugins consume them directly.
//
// Plugin name: "EncoderPreQuant", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: encoder  [B, S_enc, K_enc]  BF16
//
// Plugin fields:
//   "static_act_scale_enc"  FP32 length = max_S_enc  (OPTIONAL - if present, uses static path)
//   "K_enc"                 INT32
//
// Outputs:
//   0: encoder_i8     [B, S_enc, K_enc]  INT8
//   1: encoder_scale  [B, S_enc]         FP32
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class EncoderPreQuantPlugin : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
public:
    EncoderPreQuantPlugin(std::string const& name, int32_t KEnc);
    EncoderPreQuantPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    EncoderPreQuantPlugin() = delete;
    EncoderPreQuantPlugin(EncoderPreQuantPlugin const&) = delete;
    ~EncoderPreQuantPlugin() override;

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

    std::vector<float> mStaticActScaleEncHost;   // optional
    void*  mStaticActScaleEncDevice{nullptr};

    // FoldQuant: block-diagonal butterfly + the RAW-frame SmoothQuant vector. Zero
    // block size means the unrotated path, exactly as in the INT4 sibling;
    // there is no third state and no silent skip.
    int32_t mRotBlockSize{0};
    std::vector<float> mScalePreEncHost;
    void*  mScalePreEncDevice{nullptr};

    int32_t mKEnc{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class EncoderPreQuantPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    EncoderPreQuantPluginCreator();
    ~EncoderPreQuantPluginCreator() override = default;

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
