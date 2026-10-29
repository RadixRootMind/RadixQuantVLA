// FusedCrossAttnFullInt4: INT4 cross-attention block + FoldQuant rotation.
// Clone of FusedCrossAttnFull (INT8). The Q and attn_O prologues fold their
// rotation in-plugin; the KV path consumes the encoder that EncoderPreQuantInt4
// already permuted+block-rotated+int4-quantized once (shared encoder rotation),
// so there is no in-plugin KV rotation. Plugin name "FusedCrossAttnFullInt4".
//
// Inputs:
//   0:x [B,S,K] BF16  1:scale [B,K] BF16  2:shift [B,K] BF16
//   3:enc_i4 [B,S_enc,K_enc/8] INT32 (packed int4)  4:enc_scale [B,S_enc] FP32
//   5:attn_mask [B,1,1,S_enc] BF16 (additive; optional → present iff 6 inputs)
// Outputs: 0:out [B,S,K] BF16   (1:kv_bf16 only if emit_kv != 0)
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class FusedCrossAttnFullInt4Plugin : public nvinfer1::IPluginV3,
                                     public nvinfer1::IPluginV3OneCore,
                                     public nvinfer1::IPluginV3OneBuild,
                                     public nvinfer1::IPluginV3OneRuntime {
public:
    FusedCrossAttnFullInt4Plugin(std::string const& name,
        std::vector<int8_t> wQ,  std::vector<float> wQScale,  std::vector<float> bQ,
        std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
        std::vector<int8_t> wO,  std::vector<float> wOScale,  std::vector<float> bO,
        std::vector<int32_t> permQ, std::vector<uint16_t> rotQ,
        std::vector<int32_t> permO, std::vector<uint16_t> rotO,
        int32_t innerDim, int32_t K, int32_t KEnc,
        int32_t numHeads, int32_t headDim, int32_t blockSize, float eps);
    FusedCrossAttnFullInt4Plugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    FusedCrossAttnFullInt4Plugin() = delete;
    FusedCrossAttnFullInt4Plugin(FusedCrossAttnFullInt4Plugin const&) = delete;
    ~FusedCrossAttnFullInt4Plugin() override;

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

    std::vector<int8_t> mQI4Host, mKVI4Host, mOI4Host;
    std::vector<float>  mQScaleHost, mKVScaleHost, mOScaleHost;
    std::vector<float>  mQBiasHost,  mKVBiasHost,  mOBiasHost;
    std::vector<float> mScalePreInHost;
    std::vector<float> mScalePreOHost;
    int32_t mRotBlockSize{0};
    std::vector<int32_t> mPermQHost, mPermOHost;
    std::vector<uint16_t> mRotQHost, mRotOHost;   // BF16 raw (Lever 1: was FP32)

    void* mQI4Device{nullptr};   void* mKVI4Device{nullptr};  void* mOI4Device{nullptr};
    void* mQScaleDevice{nullptr}; void* mKVScaleDevice{nullptr}; void* mOScaleDevice{nullptr};
    void* mQBiasDevice{nullptr};  void* mKVBiasDevice{nullptr};  void* mOBiasDevice{nullptr};
    // Butterfly mode (rot_block_size > 0): no baked rotation matrix; the
    // kernel recomputes the Hadamard and the fold-before SmoothQuant
    // vector divides the raw channel on the way in.
    void const* mScalePreInDevice{nullptr};
    void const* mScalePreODevice{nullptr};
    void* mPermQDevice{nullptr};  void* mRotQBf16Device{nullptr};
    void* mPermODevice{nullptr};  void* mRotOBf16Device{nullptr};

    // Non-owning views into mShared (bound in attachToContext); freed by the resource.
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;

    void* mCublasHandle{nullptr};

    int32_t mInnerDim{0};
    int32_t mK{0};
    int32_t mKEnc{0};
    int32_t mNumHeads{0};
    int32_t mHeadDim{0};
    int32_t mBlockSize{0};
    float   mEps{1e-5f};
    bool    mHasMask{false};
    int32_t mHasMaskSerialized{0};   // staging for the has_mask serialize field
    int32_t mEmitKV{0};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedCrossAttnFullInt4PluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    FusedCrossAttnFullInt4PluginCreator();
    ~FusedCrossAttnFullInt4PluginCreator() override = default;
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
