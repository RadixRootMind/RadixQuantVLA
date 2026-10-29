// FusedFfnBlockInt4: full DiT FFN block as one TRT plugin, true W4A4 + FoldQuant
// rotation. Clone of FusedFfnBlock (INT8) with int4 weights/GEMMs and the activation
// rotation folded into each prologue.
//
//     norm3 = LayerNorm(x, elementwise_affine=False)
//     h0    = ff.net.0.proj(norm3)        // Linear(K → inner_dim, bias)
//     h1    = gelu(h0, "tanh")
//     out   = ff.net.2(h1) + x            // Linear(inner_dim → K, bias) + residual
//
// Both proj0 and proj2 inputs are permuted+block-rotated before int4 quant; the
// rotated weight is baked per-output-channel int4. Plugin name "FusedFfnBlockInt4",
// namespace "gr00t::v1", version "1".
//
// Inputs:  0: x [B,S,K] BF16     Outputs: 0: y [B,S,K] BF16 (+residual)
//
// Plugin fields (baked):
//   weight_proj0_i4  INT8 (inner_dim*K/2 packed) / weight_proj0_scale FP32 (inner_dim) / bias_proj0 FP32 (inner_dim)
//   weight_proj2_i4  INT8 (K*inner_dim/2 packed) / weight_proj2_scale FP32 (K)         / bias_proj2 FP32 (K)
//   perm0 INT32 (K)        rotation0 FP32 (K/bs * bs*bs)        - proj0 input rotation
//   perm2 INT32 (inner_dim) rotation2 FP32 (inner_dim/bs * bs*bs) - proj2 input rotation
//   inner_dim INT32, K INT32, block_size INT32, eps FP32
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedFfnBlockInt4Plugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
public:
    FusedFfnBlockInt4Plugin(
        std::string const& name,
        std::vector<int8_t> wProj0I4, std::vector<float> wProj0Scale, std::vector<float> bProj0,
        std::vector<int8_t> wProj2I4, std::vector<float> wProj2Scale, std::vector<float> bProj2,
        std::vector<int32_t> perm0, std::vector<uint16_t> rot0,
        std::vector<int32_t> perm2, std::vector<uint16_t> rot2,
        int32_t innerDim, int32_t K, int32_t blockSize, float eps);
    FusedFfnBlockInt4Plugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedFfnBlockInt4Plugin() = delete;
    FusedFfnBlockInt4Plugin(FusedFfnBlockInt4Plugin const&) = delete;
    ~FusedFfnBlockInt4Plugin() override;

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
    void ensureCublasHandle();
    // Immutable weights shared across execution contexts via an IPluginResource;
    // canonical order defined in the .cpp (see hostWeightSpecs/bindDeviceWeights).
    std::vector<WeightSpec> hostWeightSpecs() const;
    void bindDeviceWeights();

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t> mProj0I4Host;
    std::vector<float> mProj0ScaleHost;
    std::vector<float> mProj0BiasHost;
    std::vector<int8_t> mProj2I4Host;
    std::vector<float> mProj2ScaleHost;
    std::vector<float> mProj2BiasHost;
    // Butterfly mode (rot_block_size > 0): no baked rotation matrix, the kernel
    // recomputes the Hadamard and the SmoothQuant vector rides alongside. The
    // fold order here is fold-BEFORE (SmoothRot), so the scale divides the raw
    // channel ahead of the transform.
    std::vector<float> mScalePre0Host;
    std::vector<float> mScalePre2Host;
    int32_t mRotBlockSize{0};
    std::vector<int32_t> mPerm0Host;
    std::vector<uint16_t> mRot0Host;   // BF16 raw (Lever 1: was FP32)
    std::vector<int32_t> mPerm2Host;
    std::vector<uint16_t> mRot2Host;   // BF16 raw

    void* mProj0I4Device{nullptr};
    void* mProj0ScaleDevice{nullptr};
    void* mProj0BiasDevice{nullptr};
    void* mProj2I4Device{nullptr};
    void* mProj2ScaleDevice{nullptr};
    void* mProj2BiasDevice{nullptr};
    void* mPerm0Device{nullptr};
    void const* mScalePre0Device{nullptr};  // butterfly mode: fold-before SmoothQuant, site 0
    void const* mScalePre2Device{nullptr};  // butterfly mode: fold-before SmoothQuant, site 2
    void* mRot0Bf16Device{nullptr};   // R0 as BF16 (nb0*bs*bs) for the cuBLAS rotation
    void* mPerm2Device{nullptr};
    void* mRot2Bf16Device{nullptr};

    // Non-owning views into mShared (bound in attachToContext); freed by the resource.
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;

    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mBlockSize{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedFfnBlockInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedFfnBlockInt4PluginCreator();
    ~FusedFfnBlockInt4PluginCreator() override = default;

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
