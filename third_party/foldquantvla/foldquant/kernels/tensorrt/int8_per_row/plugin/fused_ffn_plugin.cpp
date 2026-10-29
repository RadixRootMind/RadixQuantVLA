// IPluginV3 wrapper for fused FFN block (LN + proj0 + GELU + proj2 + residual).

#include "plugin_field_util.h"
#include "fused_ffn_plugin.h"
#include "fused_adaln_quant.h"
#include "gelu_quant.h"
#include "dit_int8_rowwise_v2_fused.h"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedFfnBlock"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

// Workspace layout (128-byte aligned per region):
//   [act1_i8        : M * K           ]
//   [act1_scale     : M * 4           ]
//   [h0_bf16        : M * inner_dim * 2]
//   [act2_i8        : M * inner_dim   ]
//   [act2_scale     : M * 4           ]
inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t innerDim) {
    size_t a1   = alignUp(static_cast<size_t>(M) * static_cast<size_t>(K));
    size_t s1   = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t h0   = alignUp(static_cast<size_t>(M) * static_cast<size_t>(innerDim) * sizeof(uint16_t));
    size_t a2   = alignUp(static_cast<size_t>(M) * static_cast<size_t>(innerDim));
    size_t s2   = alignUp(static_cast<size_t>(M) * sizeof(float));
    return a1 + s1 + h0 + a2 + s2;
}
}  // anon

PluginFieldCollection FusedFfnBlockPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedFfnBlockPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedFfnBlockPluginCreator);

FusedFfnBlockPlugin::FusedFfnBlockPlugin(
    std::string const& name,
    std::vector<int8_t> wProj0I8, std::vector<float> wProj0Scale, std::vector<float> bProj0,
    std::vector<int8_t> wProj2I8, std::vector<float> wProj2Scale, std::vector<float> bProj2,
    int32_t innerDim, int32_t K, float eps)
    : mLayerName(name)
    , mProj0I8Host(std::move(wProj0I8))
    , mProj0ScaleHost(std::move(wProj0Scale))
    , mProj0BiasHost(std::move(bProj0))
    , mProj2I8Host(std::move(wProj2I8))
    , mProj2ScaleHost(std::move(wProj2Scale))
    , mProj2BiasHost(std::move(bProj2))
    , mInnerDim(innerDim)
    , mK(K)
    , mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedFfnBlockPlugin::FusedFfnBlockPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_proj0_i8")    { auto* p = static_cast<int8_t const*>(f.data); mProj0I8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_proj0_scale") { auto* p = static_cast<float const*>(f.data); mProj0ScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_proj0")    { auto* p = static_cast<float const*>(f.data); mProj0BiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_proj2_i8")    { auto* p = static_cast<int8_t const*>(f.data); mProj2I8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_proj2_scale") { auto* p = static_cast<float const*>(f.data); mProj2ScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_proj2")    { auto* p = static_cast<float const*>(f.data); mProj2BiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_pre")       { auto* p = static_cast<float const*>(f.data); mStaticActScalePreHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_post_gelu") { auto* p = static_cast<float const*>(f.data); mStaticActScalePostGeluHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "inner_dim")     { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")             { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")           { mEps = *static_cast<float const*>(f.data); }
        else if (n == "rot_block_size") { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre0") { auto* p = static_cast<float const*>(f.data); mScalePre0Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre2") { auto* p = static_cast<float const*>(f.data); mScalePre2Host.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
    }
}

FusedFfnBlockPlugin::~FusedFfnBlockPlugin() {
    if (mProj0I8Device)               cudaFree(mProj0I8Device);
    if (mProj0ScaleDevice)            cudaFree(mProj0ScaleDevice);
    if (mProj0BiasDevice)             cudaFree(mProj0BiasDevice);
    if (mProj2I8Device)               cudaFree(mProj2I8Device);
    if (mProj2ScaleDevice)            cudaFree(mProj2ScaleDevice);
    if (mProj2BiasDevice)             cudaFree(mProj2BiasDevice);
    if (mZeroScaleDevice)             cudaFree(mZeroScaleDevice);
    if (mZeroShiftDevice)             cudaFree(mZeroShiftDevice);
    if (mStaticActScalePreDevice)     cudaFree(mStaticActScalePreDevice);
    if (mStaticActScalePostGeluDevice)cudaFree(mStaticActScalePostGeluDevice);
    if (mScalePre0Device)             cudaFree(mScalePre0Device);
    if (mScalePre2Device)             cudaFree(mScalePre2Device);
}

void FusedFfnBlockPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mProj0I8Device,    mProj0I8Host.data(),    mProj0I8Host.size());
    upload(&mProj0ScaleDevice, mProj0ScaleHost.data(), mProj0ScaleHost.size() * sizeof(float));
    upload(&mProj0BiasDevice,  mProj0BiasHost.data(),  mProj0BiasHost.size() * sizeof(float));
    upload(&mProj2I8Device,    mProj2I8Host.data(),    mProj2I8Host.size());
    upload(&mProj2ScaleDevice, mProj2ScaleHost.data(), mProj2ScaleHost.size() * sizeof(float));
    upload(&mProj2BiasDevice,  mProj2BiasHost.data(),  mProj2BiasHost.size() * sizeof(float));
    upload(&mScalePre0Device, mScalePre0Host.data(), mScalePre0Host.size() * sizeof(float));
    upload(&mScalePre2Device, mScalePre2Host.data(), mScalePre2Host.size() * sizeof(float));
    upload(&mStaticActScalePreDevice,      mStaticActScalePreHost.data(),
           mStaticActScalePreHost.size() * sizeof(float));
    upload(&mStaticActScalePostGeluDevice, mStaticActScalePostGeluHost.data(),
           mStaticActScalePostGeluHost.size() * sizeof(float));
}

void FusedFfnBlockPlugin::ensureZeroScaleShift(int32_t B, cudaStream_t stream) {
    if (mZeroScaleDevice && mZeroBufferB == B) return;
    if (mZeroScaleDevice) cudaFree(mZeroScaleDevice);
    if (mZeroShiftDevice) cudaFree(mZeroShiftDevice);
    size_t bytes = static_cast<size_t>(B) * mK * sizeof(uint16_t);  // BF16
    cudaMalloc(&mZeroScaleDevice, bytes);
    cudaMalloc(&mZeroShiftDevice, bytes);
    cudaMemsetAsync(mZeroScaleDevice, 0, bytes, stream);
    cudaMemsetAsync(mZeroShiftDevice, 0, bytes, stream);
    mZeroBufferB = B;
}

IPluginCapability* FusedFfnBlockPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedFfnBlockPlugin::clone() noexcept {
    try {
        auto* p = new FusedFfnBlockPlugin(
            mLayerName,
            mProj0I8Host, mProj0ScaleHost, mProj0BiasHost,
            mProj2I8Host, mProj2ScaleHost, mProj2BiasHost,
            mInnerDim, mK, mEps);
        p->mStaticActScalePreHost      = mStaticActScalePreHost;
        p->mStaticActScalePostGeluHost = mStaticActScalePostGeluHost;
        // The clone is what runs; a member dropped here silently stops the fold.
        p->mRotBlockSize  = mRotBlockSize;
        p->mScalePre0Host = mScalePre0Host;
        p->mScalePre2Host = mScalePre2Host;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedFfnBlockPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedFfnBlockPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedFfnBlockPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedFfnBlockPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedFfnBlockPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedFfnBlockPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedFfnBlockPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    outputs[0] = inputs[0];  // same shape (residual is in-place semantics)
    return 0;
}

bool FusedFfnBlockPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedFfnBlockPlugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedFfnBlockPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int64_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), static_cast<int32_t>(Kmax), mInnerDim);
}

int32_t FusedFfnBlockPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        int32_t B = static_cast<int32_t>(M / S);
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);

        ensureZeroScaleShift(B, stream);

        size_t a1 = alignUp(static_cast<size_t>(M) * K);
        size_t s1 = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t h0 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(mInnerDim) * sizeof(uint16_t));
        size_t a2 = alignUp(static_cast<size_t>(M) * static_cast<size_t>(mInnerDim));
        // s2 follows but we don't need its offset variable.
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* act1_i8     = reinterpret_cast<int8_t*>(ws);
        float*  act1_scale  = reinterpret_cast<float*>(ws + a1);
        void*   h0_bf16     = static_cast<void*>(ws + a1 + s1);
        int8_t* act2_i8     = reinterpret_cast<int8_t*>(ws + a1 + s1 + h0);
        float*  act2_scale  = reinterpret_cast<float*>(ws + a1 + s1 + h0 + a2);

        // Step 1: LN + per-row INT8 quant (using fused_adaln_quant with zero scale/shift).
        int rc;
        if (mRotBlockSize > 0) {
            // FoldQuant: LN, raw-frame divide, butterfly, then the per-row amax,
            // all inside one launch, with the amax taken on the rotated row.
            rc = fused_adaln_fwht_quant_bf16_to_int8(
                inputs[0], mZeroScaleDevice, mZeroShiftDevice, mScalePre0Device,
                act1_i8, act1_scale,
                B, S, K, mEps, mRotBlockSize, stream);
        } else if (mStaticActScalePreDevice != nullptr &&
            (int)mStaticActScalePreHost.size() >= static_cast<int>(M)) {
            rc = fused_adaln_static_quant_bf16_to_int8(
                inputs[0], mZeroScaleDevice, mZeroShiftDevice,
                mStaticActScalePreDevice,
                act1_i8, act1_scale,
                B, S, K, mEps, stream);
        } else {
            rc = fused_adaln_quant_bf16_to_int8(
                inputs[0], mZeroScaleDevice, mZeroShiftDevice,
                act1_i8, act1_scale,
                B, S, K, mEps, stream);
        }
        if (rc != 0) return rc;

        // Step 2: proj0: INT8 GEMM + bias → BF16 (M, inner_dim).
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            act1_i8, mProj0I8Device,
            act1_scale, mProj0ScaleDevice, mProj0BiasDevice,
            h0_bf16,
            static_cast<int32_t>(M), mInnerDim, K, stream);
        if (rc != 0) return rc;

        // Step 3: GELU(tanh) + per-row INT8 quant of (M, inner_dim).
        if (mRotBlockSize > 0) {
            rc = dit_gelu_fwht_quant_bf16_to_int8(
                h0_bf16, mScalePre2Device, act2_i8, act2_scale,
                static_cast<int32_t>(M), mInnerDim, mRotBlockSize, stream);
        } else if (mStaticActScalePostGeluDevice != nullptr &&
            (int)mStaticActScalePostGeluHost.size() >= static_cast<int>(M)) {
            rc = dit_gelu_static_quant_bf16_to_int8(
                h0_bf16, mStaticActScalePostGeluDevice,
                act2_i8, act2_scale,
                static_cast<int32_t>(M), mInnerDim, stream);
        } else {
            rc = dit_gelu_quant_bf16_to_int8(
                h0_bf16, act2_i8, act2_scale,
                static_cast<int32_t>(M), mInnerDim, stream);
        }
        if (rc != 0) return rc;

        // Step 4: proj2: INT8 GEMM + bias + residual (residual = original x) → BF16 (M, K).
        rc = dit_int8_rowwise_gemm_bias_residual_bf16out(
            act2_i8, mProj2I8Device,
            act2_scale, mProj2ScaleDevice,
            mProj2BiasDevice, inputs[0],
            outputs[0],
            static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedFfnBlockPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedFfnBlockPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedFfnBlockPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* name, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kINT8,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* name, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kFLOAT32,
                                                  static_cast<int32_t>(v.size())));
    };
    pushI8("weight_proj0_i8",    mProj0I8Host);
    pushF32("weight_proj0_scale", mProj0ScaleHost);
    pushF32("bias_proj0",         mProj0BiasHost);
    pushI8("weight_proj2_i8",    mProj2I8Host);
    pushF32("weight_proj2_scale", mProj2ScaleHost);
    pushF32("bias_proj2",         mProj2BiasHost);
    if (!mStaticActScalePreHost.empty()) {
        pushF32("static_act_scale_pre", mStaticActScalePreHost);
    }
    if (!mStaticActScalePostGeluHost.empty()) {
        pushF32("static_act_scale_post_gelu", mStaticActScalePostGeluHost);
    }
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    if (!mScalePre0Host.empty()) {
        mDataToSerialize.emplace_back(PluginField("act_scale_pre0", mScalePre0Host.data(),
            PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePre0Host.size())));
    }
    if (!mScalePre2Host.empty()) {
        mDataToSerialize.emplace_back(PluginField("act_scale_pre2", mScalePre2Host.data(),
            PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePre2Host.size())));
    }
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// Creator -----------------------------------------------------------------

FusedFfnBlockPluginCreator::FusedFfnBlockPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_proj0_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj0_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_proj0", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj2_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj2_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_proj2", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_pre", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_post_gelu", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre0", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre2", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedFfnBlockPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedFfnBlockPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedFfnBlockPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedFfnBlockPluginCreator::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void FusedFfnBlockPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedFfnBlockPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedFfnBlockPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedFfnBlock(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedFfnBlockPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedFfnBlockPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedFfnBlockPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
