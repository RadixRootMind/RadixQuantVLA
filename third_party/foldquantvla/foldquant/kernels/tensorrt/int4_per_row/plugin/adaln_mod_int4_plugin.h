// AdaLNModInt4: AdaLN modulation linear (norm1.linear) as INT4 weight-only,
// BF16 activation (W4A16). Replaces the BF16 MatMul+Add(bias) that produces the
// modulation vector (later Split into scale/shift). The INT4 weight is stored
// ONCE (~72 MB total vs 288 MB BF16) and consumed by a fused dequant GEMV
// (dit_adaln_gemv_int4_bf16): no runtime dequant-to-BF16 copy, no extra latency
// (M = B is tiny). Accuracy: weight-only int4 on AdaLN is ~lossless (cos ~0.9916).
//
// Inputs:  0: x [.., in] BF16  (the SiLU(temb) modulation input)
// Outputs: 0: out [.., out] BF16  (= x·Wᵀ + bias, pre-Split)
// Fields:  weight_i4 INT8 (out*((in+1)/2) packed bytes, row-major [out,in]),
//          weight_scale BF16 (out), bias BF16 (out), in_dim INT32, out_dim INT32.
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class AdaLNModInt4Plugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime {
public:
    AdaLNModInt4Plugin(std::string const& name,
        std::vector<int8_t> wI4, std::vector<uint16_t> wScale, std::vector<uint16_t> bias,
        int32_t inDim, int32_t outDim, int32_t actBits = 16);
    AdaLNModInt4Plugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    AdaLNModInt4Plugin() = delete;
    AdaLNModInt4Plugin(AdaLNModInt4Plugin const&) = delete;
    ~AdaLNModInt4Plugin() override;

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
    std::vector<int8_t>   mWI4Host;     // packed int4 [out, (in+1)/2]
    std::vector<uint16_t> mWScaleHost;  // BF16 [out]
    std::vector<uint16_t> mBiasHost;    // BF16 [out]
    void* mWI4Device{nullptr};          // non-owning view into mShared
    void* mWScaleDevice{nullptr};       // non-owning view into mShared
    void* mBiasDevice{nullptr};         // non-owning view into mShared
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;
    int32_t mInDim{0};
    int32_t mOutDim{0};
    int32_t mActBits{16};   // 16 = BF16 activation (W4A16); 4 = int4 activation (W4A4)

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class AdaLNModInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    AdaLNModInt4PluginCreator();
    ~AdaLNModInt4PluginCreator() override = default;
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
