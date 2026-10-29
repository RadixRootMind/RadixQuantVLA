// FusedFfnBlock: full DiT FFN block as one TRT plugin (shared by both v1 and v2).
//
// Replaces the BasicTransformerBlock FFN sequence:
//     norm3 = LayerNorm(x, elementwise_affine=False)
//     h0    = ff.net.0.proj(norm3)        // Linear(K → inner_dim, bias)
//     h1    = F.gelu(h0, approximate="tanh")
//     out   = ff.net.2(h1)                // Linear(inner_dim → K, bias)
//     y     = out + x                     // residual
//
// Plugin name: "FusedFfnBlock", namespace "gr00t::v1", version "1".
//
// Inputs:
//   0: x  [B, S, K]  BF16
//
// Plugin fields (baked):
//   "weight_proj0_i8"    INT8  (inner_dim * K)
//   "weight_proj0_scale" FP32  (inner_dim)
//   "bias_proj0"         FP32  (inner_dim)
//   "weight_proj2_i8"    INT8  (K * inner_dim)
//   "weight_proj2_scale" FP32  (K)
//   "bias_proj2"         FP32  (K)
//   "inner_dim"          INT32
//   "K"                  INT32
//   "eps"                FP32
//
// Outputs:
//   0: y  [B, S, K]  BF16   (with + residual)
#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedFfnBlockPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
public:
    FusedFfnBlockPlugin(
        std::string const& name,
        std::vector<int8_t> wProj0I8, std::vector<float> wProj0Scale, std::vector<float> bProj0,
        std::vector<int8_t> wProj2I8, std::vector<float> wProj2Scale, std::vector<float> bProj2,
        int32_t innerDim, int32_t K, float eps);
    FusedFfnBlockPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedFfnBlockPlugin() = delete;
    FusedFfnBlockPlugin(FusedFfnBlockPlugin const&) = delete;
    ~FusedFfnBlockPlugin() override;

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
    void ensureZeroScaleShift(int32_t B, cudaStream_t stream);

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t> mProj0I8Host;
    std::vector<float> mProj0ScaleHost;
    std::vector<float> mProj0BiasHost;
    std::vector<int8_t> mProj2I8Host;
    std::vector<float> mProj2ScaleHost;
    std::vector<float> mProj2BiasHost;
    // Optional static per-row activation scales (length 0 = dynamic).
    std::vector<float> mStaticActScalePreHost;       // post-LN, M=B*S
    std::vector<float> mStaticActScalePostGeluHost;  // post-GELU, M=B*S

    // FoldQuant: the block-diagonal butterfly and the RAW-frame SmoothQuant vectors
    // for the block's two GEMM inputs (post-LN into proj0, post-GELU into proj2).
    // Zero block size is the unrotated path; there is no third state.
    int32_t mRotBlockSize{0};
    std::vector<float> mScalePre0Host;   // (K,)         post-LN channel scale
    std::vector<float> mScalePre2Host;   // (inner_dim,) post-GELU channel scale
    void* mScalePre0Device{nullptr};
    void* mScalePre2Device{nullptr};

    void* mProj0I8Device{nullptr};
    void* mProj0ScaleDevice{nullptr};
    void* mProj0BiasDevice{nullptr};
    void* mProj2I8Device{nullptr};
    void* mProj2ScaleDevice{nullptr};
    void* mProj2BiasDevice{nullptr};
    void* mStaticActScalePreDevice{nullptr};
    void* mStaticActScalePostGeluDevice{nullptr};

    // Zero buffers for the reuse-fused_adaln_quant trick (pure LN, no modulation).
    void* mZeroScaleDevice{nullptr};
    void* mZeroShiftDevice{nullptr};
    int32_t mZeroBufferB{0};

    int32_t mInnerDim{0};
    int32_t mK{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedFfnBlockPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedFfnBlockPluginCreator();
    ~FusedFfnBlockPluginCreator() override = default;

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
