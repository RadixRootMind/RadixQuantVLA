// PerRowInt4LinearResidual: FoldQuant rotated per-row INT4 quant + W4A4 Linear
// (no bias) + residual add, all in one IPluginV3.
//
// INT4 analogue of PerRowInt8LinearResidual, for linears that no macro plugin
// covers (e.g. an action-head FFN whose LayerNorm is followed by a *runtime*
// time-embedding add that FusedFfnBlockInt4's baked no-affine norm cannot
// express, so the norm/add/GELU stay ONNX ops and only the two GEMMs run
// through this plugin).
//
// Pipeline:
//   1. dit_int4_per_row_rotate_quant_bf16(x, perm, rotation) → x_i4, x_scale
//      (fused permute + block rotation + per-row INT4 quant; under the
//       w4a4_sr scheme the rotation baked here is SmoothQuant-folded
//       and stays numerically paired with the packed weight)
//   2. dit_int4_rowwise_gemm_residual_bf16out(x_i4, weight_i4, ..., residual) → y
//
// Plugin namespace: "gr00t::v1"
// Plugin name:      "PerRowInt4LinearResidual"
//
// Inputs (runtime):
//   0: x        [B, S, K]  BF16  - raw (un-rotated) pre-linear activation
//   1: residual [B, S, N]  BF16  - added to GEMM output
//
// Plugin fields (baked):
//   "weight_i4"     INT8  (N*K/2 packed nibbles, column-major - see
//                          rotation.pack_int4_colmajor{,_sq})
//   "weight_scale"  FP32  (N,)
//   "perm"          INT32 (K,)               - channel permutation (dense-rotation mode)
//   "rotation"      FP32  (K/bs * bs * bs)   - per-block rotation (dense-rotation mode)
//   "N"             INT32 scalar
//   "K"             INT32 scalar
//   "block_size"    INT32 scalar             - dense-rotation block
//   "rot_block_size" INT32 scalar            - FWHT mode (see below)
//
// Two rotation modes, selected by the baked field set (one creator, two users):
//   FoldQuant mode: "perm"+"rotation" present -> fused permute + dense per-block
//                  rotation quant (expert emitters, unchanged).
//   FWHT mode: "perm"/"rotation" absent, "rot_block_size" >= 1 -> in-plugin
//                  block-diagonal Sylvester Hadamard before per-row quant, the
//                  LLM W4A4 residual path (o_proj / down_proj). Weight side is
//                  folded offline with W·Hᵀ (llm_rotation_sq.apply_rot_fold).
//
// Outputs:
//   0: y        [B, S, N]  BF16  - INT4 GEMM result + residual
#pragma once

#include <NvInferRuntime.h>
#include "int4_weight_resource.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gr00t {
namespace v1 {
namespace plugins {

class PerRowInt4LinearResidualPlugin : public nvinfer1::IPluginV3,
                                       public nvinfer1::IPluginV3OneCore,
                                       public nvinfer1::IPluginV3OneBuild,
                                       public nvinfer1::IPluginV3OneRuntime {
public:
    PerRowInt4LinearResidualPlugin(std::string const& name,
        std::vector<int8_t> weightI4, std::vector<float> weightScale,
        std::vector<int32_t> perm, std::vector<float> rotation,
        int32_t N, int32_t K, int32_t blockSize, int32_t rotBlockSize = 0, float actClipRatio = 1.0f);
    PerRowInt4LinearResidualPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    PerRowInt4LinearResidualPlugin() = delete;
    PerRowInt4LinearResidualPlugin(PerRowInt4LinearResidualPlugin const&) = delete;
    ~PerRowInt4LinearResidualPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* ns) noexcept;

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
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* ctx) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    std::vector<WeightSpec> hostWeightSpecs() const;
    void bindDeviceWeights();
    void ensureCublasHandle();
    // BF16 copy of the FP32 rotation field, built once on the host. The plugin
    // field stays FP32 (existing engines unchanged); cuBLAS wants BF16 operands.
    void buildRotationBf16();

    std::string mLayerName;
    std::string mNamespace;

    std::vector<int8_t>  mWeightI4Host;
    std::vector<float>   mWeightScaleHost;
    std::vector<int32_t> mPermHost;
    std::vector<float>   mRotationHost;
    std::vector<uint16_t> mRotationBf16Host;
    // FWHT mode's post-rotation SmoothQuant vector (K,). Empty = no SQ.
    std::vector<float>   mActScaleChHost;
    // Fold-before (SmoothRot) twin: divides the RAW channel ahead of the
    // butterfly, where mActScaleChHost divides the rotated one after it.
    std::vector<float>   mActScalePreHost;
    int32_t mN{0};
    int32_t mK{0};
    int32_t mBlockSize{0};
    int32_t mRotBlockSize{0};
    float mActClipRatio{1.0f};  // FWHT mode; 0 when dense-rotation fields drive the rotation

    // Non-owning views into the shared weight resource (attachToContext).
    SharedDeviceWeights* mShared{nullptr};
    std::string mResourceKey;
    void const* mWeightI4Device{nullptr};
    void const* mWeightScaleDevice{nullptr};
    void const* mPermDevice{nullptr};
    void const* mRotationDevice{nullptr};
    void const* mRotationBf16Device{nullptr};
    void const* mActScaleChDevice{nullptr};
    void const* mActScalePreDevice{nullptr};
    void* mCublasHandle{nullptr};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class PerRowInt4LinearResidualPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    PerRowInt4LinearResidualPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* ns) noexcept;

    nvinfer1::IPluginV3* createPlugin(char const* name,
        nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t
