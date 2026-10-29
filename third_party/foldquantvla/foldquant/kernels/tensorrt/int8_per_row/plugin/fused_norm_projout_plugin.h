// FusedNormProjOut: collapses the DiT output head
// (norm_out + AdaLN modulation from proj_out_1 + per-row INT8 quant + INT8
//  proj_out_2 GEMM + bias) into one plugin call. Optional / exploratory: the
// four shipped schemes in profile_quantization_dit.py keep the BF16 output
// head as ONNX nodes; this plugin is kept for engines that want to fold the
// head into the macro-plugin pipeline.
//
// Math (matches AlohaDiT32WithHead.forward output head):
//   shift, scale = proj_out_1(silu(temb)).chunk(2)  ← computed in ONNX BF16 outside plugin
//   hidden       = LayerNorm(hidden) * (1 + scale) + shift
//   out          = INT8_GEMM(quant(hidden) × W_proj_out_2) + bias_proj_out_2
//
// Plugin name: "FusedNormProjOut", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: hidden   [B, S, K]    BF16
//   1: scale    [B, K]       BF16 - split from proj_out_1(silu(temb)), second half
//   2: shift    [B, K]       BF16 - split from proj_out_1(silu(temb)), first half
//
// Plugin fields:
//   "weight_proj_out_i8"      INT8  length = output_dim * K
//   "weight_proj_out_scale"   FP32  length = output_dim
//   "bias_proj_out"           FP32  length = output_dim
//   "static_act_scale_x"      FP32  length = max_S  (OPTIONAL - dynamic if absent)
//   "K"                       INT32 = hidden dim
//   "output_dim"              INT32 = proj_out_2 out_features
//   "eps"                     FP32
//
// Outputs:
//   0: out      [B, S, output_dim] BF16
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedNormProjOutPlugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
public:
    FusedNormProjOutPlugin(std::string const& name,
        std::vector<int8_t> wI8, std::vector<float> wScale, std::vector<float> bias,
        int32_t K, int32_t outputDim, float eps);
    FusedNormProjOutPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedNormProjOutPlugin() = delete;
    FusedNormProjOutPlugin(FusedNormProjOutPlugin const&) = delete;
    ~FusedNormProjOutPlugin() override;

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

    std::vector<int8_t> mWI8Host;
    std::vector<float>  mWScaleHost;
    std::vector<float>  mBiasHost;
    std::vector<float>  mStaticActScaleXHost;

    void* mWI8Device{nullptr};
    void* mWScaleDevice{nullptr};
    void* mBiasDevice{nullptr};
    void* mStaticActScaleXDevice{nullptr};

    int32_t mK{0};
    int32_t mOutputDim{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedNormProjOutPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedNormProjOutPluginCreator();
    ~FusedNormProjOutPluginCreator() override = default;

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
