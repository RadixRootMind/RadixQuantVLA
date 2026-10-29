// EncoderPreQuantInt4: once-per-forward permute + shared block-rotation + per-row
// INT4 quant of the cross-attn encoder. Emits (encoder_i4 INT32-packed, encoder_scale)
// shared by all downstream FusedCrossAttnFullInt4 blocks. Clone of EncoderPreQuant.
//
// The rotation here is the SHARED FoldQuant encoder rotation (built from the stacked KV
// weights of all cross blocks); each block's KV weight is rotated by the same matrix.
//
// Inputs:  0: encoder [B, S_enc, K_enc] BF16
// Outputs: 0: encoder_i4 [B, S_enc, K_enc/8] INT32 (packed int4, K_enc/2 bytes/row)
//          1: encoder_scale [B, S_enc] FP32
// Fields:  perm_enc INT32 (K_enc), rotation_enc FP32 (K_enc/bs*bs*bs), K_enc INT32, block_size INT32
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class EncoderPreQuantInt4Plugin : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
public:
    EncoderPreQuantInt4Plugin(std::string const& name,
        std::vector<int32_t> permEnc, std::vector<float> rotEnc, int32_t KEnc, int32_t blockSize);
    EncoderPreQuantInt4Plugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    EncoderPreQuantInt4Plugin() = delete;
    EncoderPreQuantInt4Plugin(EncoderPreQuantInt4Plugin const&) = delete;
    ~EncoderPreQuantInt4Plugin() override;

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
    // Immutable weights shared across execution contexts via an IPluginResource;
    // canonical order defined in the .cpp (see hostWeightSpecs/bindDeviceWeights).
    std::vector<WeightSpec> hostWeightSpecs() const;
    void bindDeviceWeights();

    std::string mLayerName;
    std::string mNamespace;
    // Butterfly mode: no baked matrix, the fold-before SmoothQuant vector
    // divides the raw channel and the kernel recomputes the Hadamard.
    std::vector<float> mScalePreEncHost;
    int32_t mRotBlockSize{0};
    std::vector<int32_t> mPermEncHost;
    std::vector<float> mRotEncHost;
    void* mPermEncDevice{nullptr};   // non-owning view into mShared
    void const* mScalePreEncDevice{nullptr};
    void* mRotEncDevice{nullptr};    // non-owning view into mShared
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;
    int32_t mKEnc{0};
    int32_t mBlockSize{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class EncoderPreQuantInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    EncoderPreQuantInt4PluginCreator();
    ~EncoderPreQuantInt4PluginCreator() override = default;
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
