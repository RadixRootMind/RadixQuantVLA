// IPluginV3 wrapper for fused INT4 FFN block (LN + proj0 + GELU + proj2 + residual)
// with FoldQuant rotation folded into each prologue. Clone of fused_ffn_plugin.cpp.

#include "plugin_field_util.h"
#include "fused_ffn_int4_plugin.h"
#include "dit_int4_rowwise.h"
#include "int4_host_util.h"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedFfnBlockInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

// Workspace (128-byte aligned per region). B1 rotation path adds bf16 pre/post
// rotation scratch (xp,xr) reused across the proj0 and proj2 prologues:
//   [xp : M*max(K,inner)*2][xr : M*max(K,inner)*2]
//   [act1_i4 : M*K/2][act1_scale : M*4][h0_bf16 : M*inner*2][act2_i4 : M*inner/2][act2_scale : M*4]
inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t innerDim) {
    size_t W = static_cast<size_t>(M) * (K > innerDim ? K : innerDim);
    size_t xp = alignUp(W * sizeof(uint16_t));
    size_t xr = alignUp(W * sizeof(uint16_t));
    size_t a1 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(K) / 2);
    size_t s1 = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t h0 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(innerDim) * sizeof(uint16_t));
    size_t a2 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(innerDim) / 2);
    size_t s2 = alignUp(static_cast<size_t>(M) * sizeof(float));
    return xp + xr + a1 + s1 + h0 + a2 + s2;
}
}  // anon

PluginFieldCollection FusedFfnBlockInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> FusedFfnBlockInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedFfnBlockInt4PluginCreator);

FusedFfnBlockInt4Plugin::FusedFfnBlockInt4Plugin(
    std::string const& name,
    std::vector<int8_t> wProj0I4, std::vector<float> wProj0Scale, std::vector<float> bProj0,
    std::vector<int8_t> wProj2I4, std::vector<float> wProj2Scale, std::vector<float> bProj2,
    std::vector<int32_t> perm0, std::vector<uint16_t> rot0,
    std::vector<int32_t> perm2, std::vector<uint16_t> rot2,
    int32_t innerDim, int32_t K, int32_t blockSize, float eps)
    : mLayerName(name)
    , mProj0I4Host(std::move(wProj0I4))
    , mProj0ScaleHost(std::move(wProj0Scale))
    , mProj0BiasHost(std::move(bProj0))
    , mProj2I4Host(std::move(wProj2I4))
    , mProj2ScaleHost(std::move(wProj2Scale))
    , mProj2BiasHost(std::move(bProj2))
    , mPerm0Host(std::move(perm0))
    , mRot0Host(std::move(rot0))
    , mPerm2Host(std::move(perm2))
    , mRot2Host(std::move(rot2))
    , mInnerDim(innerDim)
    , mK(K)
    , mBlockSize(blockSize)
    , mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedFfnBlockInt4Plugin::FusedFfnBlockInt4Plugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_proj0_i4")        { auto* p = static_cast<int8_t const*>(f.data); mProj0I4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_proj0_scale"){ auto* p = static_cast<float const*>(f.data); mProj0ScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_proj0")        { auto* p = static_cast<float const*>(f.data); mProj0BiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_proj2_i4")   { auto* p = static_cast<int8_t const*>(f.data); mProj2I4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_proj2_scale"){ auto* p = static_cast<float const*>(f.data); mProj2ScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_proj2")        { auto* p = static_cast<float const*>(f.data); mProj2BiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "perm0")             { auto* p = static_cast<int32_t const*>(f.data); mPerm0Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation0")         { auto* p = static_cast<uint16_t const*>(f.data); mRot0Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "perm2")             { auto* p = static_cast<int32_t const*>(f.data); mPerm2Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation2")         { auto* p = static_cast<uint16_t const*>(f.data); mRot2Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "inner_dim")         { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")                 { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "block_size")        { mBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "rot_block_size")    { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre0")    { auto* p = static_cast<float const*>(f.data); mScalePre0Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre2")    { auto* p = static_cast<float const*>(f.data); mScalePre2Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "eps")               { mEps = *static_cast<float const*>(f.data); }
    }
}

FusedFfnBlockInt4Plugin::~FusedFfnBlockInt4Plugin() {
    // The mXDevice pointers are non-owning views into the shared resource; only the
    // per-context cuBLAS handle is owned here. Release our ref on the shared weights.
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights(). Rotation
// buffers are BF16 raw (uint16), everything else is byte-copied verbatim.
enum FfnW {
    W_PROJ0_I4 = 0, W_PROJ0_SCALE, W_PROJ0_BIAS,
    W_PROJ2_I4, W_PROJ2_SCALE, W_PROJ2_BIAS,
    W_PERM0, W_ROT0, W_PERM2, W_ROT2, W_SCALE_PRE0, W_SCALE_PRE2, W_COUNT
};
}  // anon

std::vector<WeightSpec> FusedFfnBlockInt4Plugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_PROJ0_I4]    = {mProj0I4Host.data(),    mProj0I4Host.size()};
    s[W_PROJ0_SCALE] = {mProj0ScaleHost.data(), mProj0ScaleHost.size() * sizeof(float)};
    s[W_PROJ0_BIAS]  = {mProj0BiasHost.data(),  mProj0BiasHost.size()  * sizeof(float)};
    s[W_PROJ2_I4]    = {mProj2I4Host.data(),    mProj2I4Host.size()};
    s[W_PROJ2_SCALE] = {mProj2ScaleHost.data(), mProj2ScaleHost.size() * sizeof(float)};
    s[W_PROJ2_BIAS]  = {mProj2BiasHost.data(),  mProj2BiasHost.size()  * sizeof(float)};
    s[W_PERM0]       = {mPerm0Host.data(),      mPerm0Host.size()      * sizeof(int32_t)};
    s[W_ROT0]        = {mRot0Host.data(),       mRot0Host.size()       * sizeof(uint16_t)};
    s[W_PERM2]       = {mPerm2Host.data(),      mPerm2Host.size()      * sizeof(int32_t)};
    s[W_ROT2]        = {mRot2Host.data(),       mRot2Host.size()       * sizeof(uint16_t)};
    s[W_SCALE_PRE0]  = {mScalePre0Host.data(),  mScalePre0Host.size()  * sizeof(float)};
    s[W_SCALE_PRE2]  = {mScalePre2Host.data(),  mScalePre2Host.size()  * sizeof(float)};
    return s;
}

void FusedFfnBlockInt4Plugin::bindDeviceWeights() {
    mProj0I4Device    = mShared->buf(W_PROJ0_I4);
    mProj0ScaleDevice = mShared->buf(W_PROJ0_SCALE);
    mProj0BiasDevice  = mShared->buf(W_PROJ0_BIAS);
    mProj2I4Device    = mShared->buf(W_PROJ2_I4);
    mProj2ScaleDevice = mShared->buf(W_PROJ2_SCALE);
    mProj2BiasDevice  = mShared->buf(W_PROJ2_BIAS);
    mPerm0Device      = mShared->buf(W_PERM0);
    mRot0Bf16Device   = mShared->buf(W_ROT0);
    mPerm2Device      = mShared->buf(W_PERM2);
    mRot2Bf16Device   = mShared->buf(W_ROT2);
    mScalePre0Device  = mScalePre0Host.empty() ? nullptr : mShared->buf(W_SCALE_PRE0);
    mScalePre2Device  = mScalePre2Host.empty() ? nullptr : mShared->buf(W_SCALE_PRE2);
}

void FusedFfnBlockInt4Plugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedFfnBlockInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedFfnBlockInt4Plugin::clone() noexcept {
    try {
        auto* p = new FusedFfnBlockInt4Plugin(
            mLayerName,
            mProj0I4Host, mProj0ScaleHost, mProj0BiasHost,
            mProj2I4Host, mProj2ScaleHost, mProj2BiasHost,
            mPerm0Host, mRot0Host, mPerm2Host, mRot2Host,
            mInnerDim, mK, mBlockSize, mEps);
        // attachToContext() clones; a member missing here is gone at runtime.
        p->mRotBlockSize = mRotBlockSize;
        p->mScalePre0Host = mScalePre0Host;
        p->mScalePre2Host = mScalePre2Host;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedFfnBlockInt4Plugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedFfnBlockInt4Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedFfnBlockInt4Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedFfnBlockInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedFfnBlockInt4Plugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedFfnBlockInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedFfnBlockInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /*exprBuilder*/) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    outputs[0] = inputs[0];
    return 0;
}

bool FusedFfnBlockInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedFfnBlockInt4Plugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedFfnBlockInt4Plugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int64_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), static_cast<int32_t>(Kmax), mInnerDim);
}

int32_t FusedFfnBlockInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        if (mShared == nullptr) return -1;  // weights bound in attachToContext
        ensureCublasHandle();

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        int32_t B = static_cast<int32_t>(M / S);
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);

        size_t W = static_cast<size_t>(M) * (K > mInnerDim ? K : mInnerDim);
        size_t xp_sz = alignUp(W * sizeof(uint16_t));
        size_t xr_sz = alignUp(W * sizeof(uint16_t));
        size_t a1 = alignUp(static_cast<size_t>(M) * K / 2);
        size_t s1 = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t h0 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(mInnerDim) * sizeof(uint16_t));
        size_t a2 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(mInnerDim) / 2);
        auto* ws = static_cast<uint8_t*>(workspace);
        void*   xp         = static_cast<void*>(ws);
        void*   xr         = static_cast<void*>(ws + xp_sz);
        uint8_t* rest      = ws + xp_sz + xr_sz;
        int8_t* act1_i4    = reinterpret_cast<int8_t*>(rest);
        float*  act1_scale = reinterpret_cast<float*>(rest + a1);
        void*   h0_bf16    = static_cast<void*>(rest + a1 + s1);
        int8_t* act2_i4    = reinterpret_cast<int8_t*>(rest + a1 + s1 + h0);
        float*  act2_scale = reinterpret_cast<float*>(rest + a1 + s1 + h0 + a2);

        // proj0 prologue: LN(no affine)+permute → cuBLAS block-rotate → INT4 quant.
        int rc = dit_adaln_permute_bf16(
            inputs[0], nullptr, nullptr, mPerm0Device, xp, B, S, K, mEps, stream);
        if (rc != 0) return rc;
        if (mRotBlockSize > 0) {
            // Butterfly: the cuBLAS block GEMM and its baked matrix are gone; the
            // Hadamard is recomputed inside the quant kernel and the SmoothQuant
            // vector divides the raw channel on the way in (fold-before order).
            rc = dit_int4_per_row_quant_fwht_bf16(xp, mScalePre0Device, nullptr,
                                                  act1_i4, act1_scale,
                                                  static_cast<int32_t>(M), K, mRotBlockSize,
                                                  /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        } else {
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xp, mRot0Bf16Device, xr,
                                            static_cast<int32_t>(M), K, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xr, act1_i4, act1_scale,
                                             static_cast<int32_t>(M), K, /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        }

        // proj0 INT4 GEMM + bias → BF16 (M, inner_dim).
        rc = dit_int4_rowwise_gemm_bias_bf16out(
            act1_i4, mProj0I4Device, act1_scale, mProj0ScaleDevice, mProj0BiasDevice,
            h0_bf16, static_cast<int32_t>(M), mInnerDim, K, stream);
        if (rc != 0) return rc;

        // proj2 prologue: GELU+permute → cuBLAS block-rotate → INT4 quant.
        rc = dit_gelu_permute_bf16(h0_bf16, mPerm2Device, xp,
                                   static_cast<int32_t>(M), mInnerDim, stream);
        if (rc != 0) return rc;
        if (mRotBlockSize > 0) {
            rc = dit_int4_per_row_quant_fwht_bf16(xp, mScalePre2Device, nullptr,
                                                  act2_i4, act2_scale,
                                                  static_cast<int32_t>(M), mInnerDim, mRotBlockSize,
                                                  /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        } else {
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xp, mRot2Bf16Device, xr,
                                            static_cast<int32_t>(M), mInnerDim, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xr, act2_i4, act2_scale,
                                             static_cast<int32_t>(M), mInnerDim, /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        }

        // proj2 INT4 GEMM + bias + residual (= original x) → BF16 (M, K).
        rc = dit_int4_rowwise_gemm_bias_residual_bf16out(
            act2_i4, mProj2I4Device, act2_scale, mProj2ScaleDevice, mProj2BiasDevice,
            inputs[0], outputs[0], static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedFfnBlockInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedFfnBlockInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<FusedFfnBlockInt4Plugin*>(clone());
        if (p == nullptr) return nullptr;
        auto specs = p->hostWeightSpecs();
        std::string key = weightDigest(kPLUGIN_NAME, specs);
        auto* r = acquireSharedWeights(key, std::move(specs));
        if (r == nullptr) { delete p; return nullptr; }
        p->mShared = r;
        p->mResourceKey = std::move(key);
        p->bindDeviceWeights();
        return p;
    } catch (...) { return nullptr; }
}

PluginFieldCollection const* FusedFfnBlockInt4Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* name, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kINT8,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* name, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kFLOAT32,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushI32 = [&](const char* name, std::vector<int32_t>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kINT32,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushBf16 = [&](const char* name, std::vector<uint16_t>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kBF16,
                                                  static_cast<int32_t>(v.size())));
    };
    pushI8("weight_proj0_i4",    mProj0I4Host);
    pushF32("weight_proj0_scale", mProj0ScaleHost);
    pushF32("bias_proj0",         mProj0BiasHost);
    pushI8("weight_proj2_i4",    mProj2I4Host);
    pushF32("weight_proj2_scale", mProj2ScaleHost);
    pushF32("bias_proj2",         mProj2BiasHost);
    pushI32("perm0", mPerm0Host);
    pushBf16("rotation0", mRot0Host);
    pushI32("perm2", mPerm2Host);
    pushBf16("rotation2", mRot2Host);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("block_size", &mBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1));
    // Butterfly mode must survive serialize: the weights are folded with W·Hᵀ,
    // so an engine that lost rot_block_size would run un-rotated and silent.
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre0", mScalePre0Host.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePre0Host.size())));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre2", mScalePre2Host.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePre2Host.size())));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// Creator -----------------------------------------------------------------

FusedFfnBlockInt4PluginCreator::FusedFfnBlockInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_proj0_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj0_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_proj0", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj2_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj2_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_proj2", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("perm0", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation0", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("perm2", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation2", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    // The ONNX parser matches a node against this list; an undeclared attribute
    // fails the whole creator lookup as "Plugin not found".
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre0", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre2", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedFfnBlockInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedFfnBlockInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedFfnBlockInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedFfnBlockInt4PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedFfnBlockInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedFfnBlockInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedFfnBlockInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedFfnBlockInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedFfnBlockInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedFfnBlockInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedFfnBlockInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
