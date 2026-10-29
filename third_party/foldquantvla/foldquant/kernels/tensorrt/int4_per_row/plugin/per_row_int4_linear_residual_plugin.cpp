// IPluginV3 wrapper: rotated per-row INT4 quant + W4A4 Linear (no bias) + residual.

#include "plugin_field_util.h"
#include "per_row_int4_linear_residual_plugin.h"
#include "dit_int4_rowwise.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"PerRowInt4LinearResidual"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp128(size_t n) { return (n + 127) & ~static_cast<size_t>(127); }

inline size_t workspaceBytes(int64_t M, int32_t K, bool dense_rotation) {
    // packed int4 activation (M, K/2 bytes) + per-row FP32 scale (M,). dense-rotation mode
    // adds two BF16 staging buffers (permuted x, rotated x) for the cuBLAS
    // rotation; FWHT mode asks for exactly what it always did, so engines built
    // before that path existed keep a valid memory plan.
    size_t a = alignUp128(static_cast<size_t>(M) * (K / 2));
    size_t s = alignUp128(static_cast<size_t>(M) * sizeof(float));
    if (!dense_rotation) return a + s;
    size_t x = alignUp128(static_cast<size_t>(M) * K * sizeof(uint16_t));
    return a + s + 2 * x;
}

// Round-to-nearest-even FP32 -> BF16, on the host (no CUDA host intrinsics).
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);  // NaN
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
}  // anon

PluginFieldCollection PerRowInt4LinearResidualPluginCreator::mFieldCollection{};
std::vector<PluginField> PerRowInt4LinearResidualPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(PerRowInt4LinearResidualPluginCreator);

PerRowInt4LinearResidualPlugin::PerRowInt4LinearResidualPlugin(std::string const& name,
    std::vector<int8_t> weightI4, std::vector<float> weightScale,
    std::vector<int32_t> perm, std::vector<float> rotation,
    int32_t N, int32_t K, int32_t blockSize, int32_t rotBlockSize, float actClipRatio)
    : mLayerName(name)
    , mWeightI4Host(std::move(weightI4))
    , mWeightScaleHost(std::move(weightScale))
    , mPermHost(std::move(perm))
    , mRotationHost(std::move(rotation))
    , mN(N), mK(K), mBlockSize(blockSize), mRotBlockSize(rotBlockSize), mActClipRatio(actClipRatio) {
    mNamespace = kPLUGIN_NAMESPACE;
    buildRotationBf16();
}


PerRowInt4LinearResidualPlugin::PerRowInt4LinearResidualPlugin(
    std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_i4")         { auto* p = static_cast<int8_t const*>(f.data);  mWeightI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_scale") { auto* p = static_cast<float const*>(f.data);   mWeightScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "perm")         { auto* p = static_cast<int32_t const*>(f.data); mPermHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation")     { auto* p = static_cast<float const*>(f.data);   mRotationHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_ch") { auto* p = static_cast<float const*>(f.data);   mActScaleChHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre"){ auto* p = static_cast<float const*>(f.data);   mActScalePreHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "N")            { mN = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")            { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "block_size")   { mBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "rot_block_size") { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_clip_ratio") { mActClipRatio = *static_cast<float const*>(f.data); }
    }
    // 1-byte payload length is ambiguous across parser builds (see
    // fieldElemCount); the packed-weight size is fully determined by N and K,
    // so trim a trailing parser NUL once both are known.
    if (mN > 0 && mK > 0 && mWeightI4Host.size() == static_cast<size_t>(mN) * (mK / 2) + 1) {
        mWeightI4Host.pop_back();
    }
    buildRotationBf16();
}

void PerRowInt4LinearResidualPlugin::buildRotationBf16() {
    mRotationBf16Host.resize(mRotationHost.size());
    for (size_t i = 0; i < mRotationHost.size(); ++i) {
        mRotationBf16Host[i] = f32_to_bf16(mRotationHost[i]);
    }
}

void PerRowInt4LinearResidualPlugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return;
    mCublasHandle = reinterpret_cast<void*>(handle);
}

PerRowInt4LinearResidualPlugin::~PerRowInt4LinearResidualPlugin() {
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights().
enum LinResW { W_WEIGHT_I4 = 0, W_WEIGHT_SCALE, W_PERM, W_ROTATION, W_ROTATION_BF16, W_ACT_SCALE_CH,
               W_ACT_SCALE_PRE, W_COUNT };
}  // anon

std::vector<WeightSpec> PerRowInt4LinearResidualPlugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_WEIGHT_I4]    = {mWeightI4Host.data(),    mWeightI4Host.size()};
    s[W_WEIGHT_SCALE] = {mWeightScaleHost.data(), mWeightScaleHost.size() * sizeof(float)};
    s[W_PERM]         = {mPermHost.data(),        mPermHost.size() * sizeof(int32_t)};
    s[W_ROTATION]     = {mRotationHost.data(),    mRotationHost.size() * sizeof(float)};
    s[W_ROTATION_BF16] = {mRotationBf16Host.data(), mRotationBf16Host.size() * sizeof(uint16_t)};
    s[W_ACT_SCALE_CH]  = {mActScaleChHost.data(),   mActScaleChHost.size() * sizeof(float)};
    s[W_ACT_SCALE_PRE] = {mActScalePreHost.data(),  mActScalePreHost.size() * sizeof(float)};
    return s;
}

void PerRowInt4LinearResidualPlugin::bindDeviceWeights() {
    mWeightI4Device    = mShared->buf(W_WEIGHT_I4);
    mWeightScaleDevice = mShared->buf(W_WEIGHT_SCALE);
    mPermDevice        = mShared->buf(W_PERM);
    mRotationDevice    = mShared->buf(W_ROTATION);
    mRotationBf16Device = mShared->buf(W_ROTATION_BF16);
    mActScaleChDevice   = mActScaleChHost.empty() ? nullptr : mShared->buf(W_ACT_SCALE_CH);
    mActScalePreDevice  = mActScalePreHost.empty() ? nullptr : mShared->buf(W_ACT_SCALE_PRE);
}

IPluginCapability* PerRowInt4LinearResidualPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* PerRowInt4LinearResidualPlugin::clone() noexcept {
    try {
        auto* p = new PerRowInt4LinearResidualPlugin(mLayerName,
            mWeightI4Host, mWeightScaleHost, mPermHost, mRotationHost, mN, mK, mBlockSize,
            mRotBlockSize, mActClipRatio);
        // attachToContext() clones, so anything not carried here is lost at
        // runtime, and a dropped SmoothQuant vector quantizes un-smoothed with
        // no visible symptom, which is exactly what the weights were folded for.
        p->mActScaleChHost = mActScaleChHost;
        p->mActScalePreHost = mActScalePreHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* PerRowInt4LinearResidualPlugin::getPluginName() const noexcept     { return kPLUGIN_NAME; }
char const* PerRowInt4LinearResidualPlugin::getPluginVersion() const noexcept  { return kPLUGIN_VERSION; }
char const* PerRowInt4LinearResidualPlugin::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void PerRowInt4LinearResidualPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t PerRowInt4LinearResidualPlugin::getNbOutputs() const noexcept { return 1; }

int32_t PerRowInt4LinearResidualPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t PerRowInt4LinearResidualPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    // Output shape = residual shape (= [B, S, N]).
    outputs[0].nbDims = inputs[1].nbDims;
    for (int32_t i = 0; i < inputs[1].nbDims - 1; ++i) {
        outputs[0].d[i] = inputs[1].d[i];
    }
    outputs[0].d[inputs[1].nbDims - 1] = exprBuilder.constant(mN);
    return 0;
}

bool PerRowInt4LinearResidualPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t PerRowInt4LinearResidualPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t PerRowInt4LinearResidualPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(Mmax, Kmax, !mRotationHost.empty());
}

int32_t PerRowInt4LinearResidualPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        if (mShared == nullptr) return -1;  // weights bound in attachToContext

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);

        size_t aBytes = alignUp128(static_cast<size_t>(M) * (K / 2));
        size_t sBytes = alignUp128(static_cast<size_t>(M) * sizeof(float));
        size_t xBytes = alignUp128(static_cast<size_t>(M) * K * sizeof(uint16_t));
        auto* ws = static_cast<uint8_t*>(workspace);
        void* actI4     = ws;
        float* actScale = reinterpret_cast<float*>(ws + aBytes);
        void* xPerm     = ws + aBytes + sBytes;           // BF16 (M, K) permuted
        void* xRot      = ws + aBytes + sBytes + xBytes;  // BF16 (M, K) rotated

        // Step 1: per-row INT4 quant, rotation mode picked by the baked fields.
        // FoldQuant mode (experts): fused permute + dense per-block
        // rotation. FWHT mode (LLM W4A4 o/down sites): block-diagonal Sylvester
        // Hadamard, weight side folded offline with W·Hᵀ. An engine baked with
        // NEITHER field set quantizes un-rotated (rot_bs <= 1 delegates).
        //
        // Fail LOUD on a mis-baked dense-rotation field set: perm-without-rotation would
        // dereference a null device pointer inside the kernel, and block_size
        // <= 0 makes the rotate-quant launcher an empty *success* that leaves
        // the workspace uninitialized, and the GEMM would then emit confident
        // garbage with no symptom. Mixing dense-rotation fields with an FWHT rot_block_size
        // is contradictory (two different rotations for one baked weight).
        int rc;
        if (!mPermHost.empty() || !mRotationHost.empty()) {
            const bool dense_rotation_complete =
                !mPermHost.empty() && !mRotationHost.empty() && mBlockSize > 0 &&
                static_cast<int32_t>(mPermHost.size()) == mK &&
                static_cast<int32_t>(mRotationHost.size()) == (mK / mBlockSize) * mBlockSize * mBlockSize &&
                mRotBlockSize <= 1;
            if (!dense_rotation_complete) return -2;
            if (mCublasHandle == nullptr) return -3;
            // permute -> cuBLAS strided-batched block rotation -> per-row quant.
            // The single fused kernel this replaces launched grid(M) blocks (10
            // for a Pi0.5 action chunk) and walked the rotation matrix out of
            // global memory twice per row, which cost 25x this composition at
            // K=4096. Same decomposition the DiT's fused attention/FFN plugins
            // already use; BF16 operands with FP32 accumulate.
            rc = dit_permute_bf16(inputs[0], mPermDevice, xPerm,
                                  static_cast<int32_t>(M), K, stream);
            if (rc != 0) return rc;
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xPerm, mRotationBf16Device, xRot,
                                            static_cast<int32_t>(M), K, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xRot, actI4, actScale,
                                             static_cast<int32_t>(M), K,
                                             /*act_clip=*/1.0f, stream);
        } else {
            // A baked act_scale_ch must match K, or the kernel would read past
            // the vector and quantize against garbage with no visible symptom.
            if (!mActScaleChHost.empty() && static_cast<int32_t>(mActScaleChHost.size()) != mK) return -4;
            if (!mActScalePreHost.empty() && static_cast<int32_t>(mActScalePreHost.size()) != mK) return -5;
            // One order or the other, never both: they fold the same scale onto
            // opposite axes and applying two would double-count it.
            if (!mActScaleChHost.empty() && !mActScalePreHost.empty()) return -6;
            rc = dit_int4_per_row_quant_fwht_bf16(
                inputs[0], mActScalePreDevice, mActScaleChDevice,
                actI4, actScale,
                static_cast<int32_t>(M), K, mRotBlockSize, mActClipRatio, stream);
        }
        if (rc != 0) return rc;

        // Step 2: fused INT4 GEMM + residual (no bias).
        rc = dit_int4_rowwise_gemm_residual_bf16out(
            actI4, mWeightI4Device,
            actScale, mWeightScaleDevice,
            inputs[1],  // residual
            outputs[0],
            static_cast<int32_t>(M), mN, K, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t PerRowInt4LinearResidualPlugin::onShapeChange(PluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* PerRowInt4LinearResidualPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<PerRowInt4LinearResidualPlugin*>(clone());
        if (p == nullptr) return nullptr;
        auto specs = p->hostWeightSpecs();
        std::string key = weightDigest(kPLUGIN_NAME, specs);
        auto* r = acquireSharedWeights(key, std::move(specs));
        if (r == nullptr) { delete p; return nullptr; }
        p->mShared = r;
        p->mResourceKey = std::move(key);
        p->bindDeviceWeights();
        p->ensureCublasHandle();
        return p;
    } catch (...) { return nullptr; }
}

PluginFieldCollection const* PerRowInt4LinearResidualPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("weight_i4", mWeightI4Host.data(),
        PluginFieldType::kINT8, static_cast<int32_t>(mWeightI4Host.size())));
    mDataToSerialize.emplace_back(PluginField("weight_scale", mWeightScaleHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mWeightScaleHost.size())));
    mDataToSerialize.emplace_back(PluginField("perm", mPermHost.data(),
        PluginFieldType::kINT32, static_cast<int32_t>(mPermHost.size())));
    mDataToSerialize.emplace_back(PluginField("rotation", mRotationHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mRotationHost.size())));
    // FWHT + SmoothQuant: the butterfly has no coefficients to absorb the
    // per-channel scale the dense rotation folds away, so it ships beside it.
    mDataToSerialize.emplace_back(PluginField("act_scale_ch", mActScaleChHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mActScaleChHost.size())));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre", mActScalePreHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mActScalePreHost.size())));
    mDataToSerialize.emplace_back(PluginField("N", &mN, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("block_size", &mBlockSize, PluginFieldType::kINT32, 1));
    // FWHT mode must survive engine serialize exactly like the dense-rotation fields: the
    // baked weights are already folded with W·Hᵀ and a deserialized engine that
    // lost rot_block_size would run un-rotated with no visible symptom.
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    // Same survival argument: a lost clip ratio silently changes the activation
    // grid the GPTQ weights were rounded against.
    mDataToSerialize.emplace_back(
        PluginField("act_clip_ratio", &mActClipRatio, PluginFieldType::kFLOAT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// Creator
// ============================================================================

PerRowInt4LinearResidualPluginCreator::PerRowInt4LinearResidualPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("perm", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation", nullptr, PluginFieldType::kFLOAT32, 0));
    // The ONNX parser matches a node against this list; an attribute the creator
    // does not declare makes the whole lookup fail as "Plugin not found".
    mPluginAttributes.emplace_back(PluginField("act_scale_ch", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_clip_ratio", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* PerRowInt4LinearResidualPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* PerRowInt4LinearResidualPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* PerRowInt4LinearResidualPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* PerRowInt4LinearResidualPluginCreator::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void PerRowInt4LinearResidualPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* PerRowInt4LinearResidualPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new PerRowInt4LinearResidualPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterPerRowInt4LinearResidual(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::PerRowInt4LinearResidualPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initPerRowInt4LinearResidualPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::PerRowInt4LinearResidualPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
