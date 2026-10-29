// FusedSelfAttnFullInt4: AdaLN + INT4 merged-QKV + cuBLAS BF16 SDPA + INT4 attn_O
// + bias + residual, with FoldQuant rotation folded into the QKV and O prologues.
// Clone of FusedSelfAttnFull (INT8). Plugin name "FusedSelfAttnFullInt4".
//
// Inputs:  0:x [B,S,K] BF16   1:scale [B,K] BF16   2:shift [B,K] BF16
// Output:  0:out [B,S,K] BF16 (+residual)
//
// Fields: weight_qkv_i4 (3*inner*K/2) / weight_qkv_scale (3*inner) / bias_qkv (3*inner);
//         weight_o_i4 (K*inner/2) / weight_o_scale (K) / bias_o (K);
//         perm_qkv INT32 (K) / rotation_qkv FP32 (K/bs*bs*bs)  - QKV input (shared Q/K/V);
//         perm_o   INT32 (inner) / rotation_o FP32 (inner/bs*bs*bs) - attn_O input;
//         inner_dim, K, num_heads, head_dim, block_size INT32; eps FP32.
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedSelfAttnFullInt4Plugin : public nvinfer1::IPluginV3,
                                    public nvinfer1::IPluginV3OneCore,
                                    public nvinfer1::IPluginV3OneBuild,
                                    public nvinfer1::IPluginV3OneRuntime {
public:
    FusedSelfAttnFullInt4Plugin(std::string const& name,
        std::vector<int8_t> wQKV, std::vector<float> wQKVScale, std::vector<float> bQKV,
        std::vector<int8_t> wO,   std::vector<float> wOScale,   std::vector<float> bO,
        std::vector<int32_t> permQKV, std::vector<uint16_t> rotQKV,
        std::vector<int32_t> permO,   std::vector<uint16_t> rotO,
        int32_t innerDim, int32_t K, int32_t numHeads, int32_t headDim,
        int32_t blockSize, float eps);
    FusedSelfAttnFullInt4Plugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedSelfAttnFullInt4Plugin() = delete;
    FusedSelfAttnFullInt4Plugin(FusedSelfAttnFullInt4Plugin const&) = delete;
    ~FusedSelfAttnFullInt4Plugin() override;

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
    // Immutable weights are shared across execution contexts via an IPluginResource.
    // hostWeightSpecs() lists the host buffers in the canonical order the device
    // views below are bound to; bindDeviceWeights() points them at the shared copy.
    std::vector<WeightSpec> hostWeightSpecs() const;
    void bindDeviceWeights();

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t> mQKVI4Host;
    std::vector<float> mQKVScaleHost;
    std::vector<float> mQKVBiasHost;
    std::vector<int8_t> mOI4Host;
    std::vector<float> mOScaleHost;
    std::vector<float> mOBiasHost;
    std::vector<float> mScalePreInHost;
    std::vector<float> mScalePreOHost;
    int32_t mRotBlockSize{0};
    std::vector<int32_t> mPermQKVHost;
    std::vector<uint16_t> mRotQKVHost;   // BF16 raw (Lever 1: was FP32)
    std::vector<int32_t> mPermOHost;
    std::vector<uint16_t> mRotOHost;     // BF16 raw

    void* mQKVI4Device{nullptr};
    void* mQKVScaleDevice{nullptr};
    void* mQKVBiasDevice{nullptr};
    void* mOI4Device{nullptr};
    void* mOScaleDevice{nullptr};
    void* mOBiasDevice{nullptr};
    void* mPermQKVDevice{nullptr};
    // Butterfly mode (rot_block_size > 0): no baked rotation matrix; the
    // kernel recomputes the Hadamard and the fold-before SmoothQuant
    // vector divides the raw channel on the way in.
    void const* mScalePreInDevice{nullptr};
    void const* mScalePreODevice{nullptr};
    void* mRotQKVBf16Device{nullptr};   // BF16 R for the cuBLAS rotation
    void* mPermODevice{nullptr};
    void* mRotOBf16Device{nullptr};

    // Non-owning views into mShared (bound in attachToContext); freed by the resource.
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;

    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mNumHeads{0};
    int32_t mHeadDim{0};
    int32_t mBlockSize{0};
    float   mEps{1e-5f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedSelfAttnFullInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedSelfAttnFullInt4PluginCreator();
    ~FusedSelfAttnFullInt4PluginCreator() override = default;
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
