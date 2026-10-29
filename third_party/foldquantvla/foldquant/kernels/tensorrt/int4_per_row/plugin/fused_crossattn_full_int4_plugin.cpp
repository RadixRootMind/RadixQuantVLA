// FusedCrossAttnFullInt4: INT4 cross-attn + FoldQuant rotation. Clone of
// fused_crossattn_full_plugin.cpp. Encoder arrives pre-rotated+int4-quantized
// (EncoderPreQuantInt4); Q and O prologues rotate in-plugin.

#include "plugin_field_util.h"
#include "fused_crossattn_full_int4_plugin.h"
#include "dit_int4_rowwise.h"
#include "sdpa_cublas.h"
#include "int4_host_util.h"

#include <cassert>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedCrossAttnFullInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

inline size_t workspaceBytes(int32_t M, int32_t K, int32_t MEnc, int32_t innerDim,
                             int32_t numHeads, int32_t S) {
    size_t Wm = static_cast<size_t>(M) * (K > innerDim ? K : innerDim);
    size_t xp = alignUp(Wm * sizeof(uint16_t));
    size_t xr = alignUp(Wm * sizeof(uint16_t));
    size_t x_i4    = alignUp(static_cast<size_t>(M) * K / 2);
    size_t x_sc    = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t q_bf16  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t kv_bf16 = alignUp(static_cast<size_t>(MEnc) * 2 * innerDim * sizeof(uint16_t));
    size_t scores  = alignUp(static_cast<size_t>(numHeads) * static_cast<size_t>(S) * MEnc * sizeof(uint16_t));
    size_t attn_b  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t attn_i4 = alignUp(static_cast<size_t>(M) * innerDim / 2);
    size_t attn_sc = alignUp(static_cast<size_t>(M) * sizeof(float));
    return xp + xr + x_i4 + x_sc + q_bf16 + kv_bf16 + scores + attn_b + attn_i4 + attn_sc;
}
}  // anon

PluginFieldCollection FusedCrossAttnFullInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> FusedCrossAttnFullInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedCrossAttnFullInt4PluginCreator);

FusedCrossAttnFullInt4Plugin::FusedCrossAttnFullInt4Plugin(std::string const& name,
    std::vector<int8_t> wQ,  std::vector<float> wQScale,  std::vector<float> bQ,
    std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
    std::vector<int8_t> wO,  std::vector<float> wOScale,  std::vector<float> bO,
    std::vector<int32_t> permQ, std::vector<uint16_t> rotQ,
    std::vector<int32_t> permO, std::vector<uint16_t> rotO,
    int32_t innerDim, int32_t K, int32_t KEnc, int32_t numHeads, int32_t headDim,
    int32_t blockSize, float eps)
    : mLayerName(name)
    , mQI4Host(std::move(wQ)), mKVI4Host(std::move(wKV)), mOI4Host(std::move(wO))
    , mQScaleHost(std::move(wQScale)), mKVScaleHost(std::move(wKVScale)), mOScaleHost(std::move(wOScale))
    , mQBiasHost(std::move(bQ)), mKVBiasHost(std::move(bKV)), mOBiasHost(std::move(bO))
    , mPermQHost(std::move(permQ)), mPermOHost(std::move(permO))
    , mRotQHost(std::move(rotQ)), mRotOHost(std::move(rotO))
    , mInnerDim(innerDim), mK(K), mKEnc(KEnc), mNumHeads(numHeads), mHeadDim(headDim)
    , mBlockSize(blockSize), mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedCrossAttnFullInt4Plugin::FusedCrossAttnFullInt4Plugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_q_i4")        { auto* p = static_cast<int8_t const*>(f.data); mQI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_q_scale"){ auto* p = static_cast<float const*>(f.data); mQScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_q")        { auto* p = static_cast<float const*>(f.data); mQBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_kv_i4")  { auto* p = static_cast<int8_t const*>(f.data); mKVI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_kv_scale"){auto* p = static_cast<float const*>(f.data); mKVScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_kv")       { auto* p = static_cast<float const*>(f.data); mKVBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_o_i4")   { auto* p = static_cast<int8_t const*>(f.data); mOI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_o_scale"){ auto* p = static_cast<float const*>(f.data); mOScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_o")        { auto* p = static_cast<float const*>(f.data); mOBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "perm_q")        { auto* p = static_cast<int32_t const*>(f.data); mPermQHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation_q")    { auto* p = static_cast<uint16_t const*>(f.data); mRotQHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "perm_o")        { auto* p = static_cast<int32_t const*>(f.data); mPermOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation_o")    { auto* p = static_cast<uint16_t const*>(f.data); mRotOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "inner_dim")     { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")             { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "K_enc")         { mKEnc = *static_cast<int32_t const*>(f.data); }
        else if (n == "num_heads")     { mNumHeads = *static_cast<int32_t const*>(f.data); }
        else if (n == "head_dim")      { mHeadDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "block_size")    { mBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "rot_block_size")   { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre_in") { auto* p = static_cast<float const*>(f.data); mScalePreInHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre_o")  { auto* p = static_cast<float const*>(f.data); mScalePreOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "eps")           { mEps = *static_cast<float const*>(f.data); }
        else if (n == "emit_kv")       { mEmitKV = *static_cast<int32_t const*>(f.data); }
        else if (n == "has_mask")      { mHasMask = (*static_cast<int32_t const*>(f.data)) != 0; }
    }
}

FusedCrossAttnFullInt4Plugin::~FusedCrossAttnFullInt4Plugin() {
    // The mXDevice pointers are non-owning views into the shared resource; only the
    // per-context cuBLAS handle is owned here. Release our ref on the shared weights.
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights(). Rotation
// buffers are BF16 raw (uint16), everything else is byte-copied verbatim.
enum CrossAttnW {
    W_Q_I4 = 0, W_Q_SCALE, W_Q_BIAS,
    W_KV_I4, W_KV_SCALE, W_KV_BIAS,
    W_O_I4, W_O_SCALE, W_O_BIAS,
    W_PERM_Q, W_ROT_Q, W_PERM_O, W_ROT_O, W_SCALE_PRE_IN, W_SCALE_PRE_O, W_COUNT
};
}  // anon

std::vector<WeightSpec> FusedCrossAttnFullInt4Plugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_Q_I4]     = {mQI4Host.data(),     mQI4Host.size()};
    s[W_Q_SCALE]  = {mQScaleHost.data(),  mQScaleHost.size()  * sizeof(float)};
    s[W_Q_BIAS]   = {mQBiasHost.data(),   mQBiasHost.size()   * sizeof(float)};
    s[W_KV_I4]    = {mKVI4Host.data(),    mKVI4Host.size()};
    s[W_KV_SCALE] = {mKVScaleHost.data(), mKVScaleHost.size() * sizeof(float)};
    s[W_KV_BIAS]  = {mKVBiasHost.data(),  mKVBiasHost.size()  * sizeof(float)};
    s[W_O_I4]     = {mOI4Host.data(),     mOI4Host.size()};
    s[W_O_SCALE]  = {mOScaleHost.data(),  mOScaleHost.size()  * sizeof(float)};
    s[W_O_BIAS]   = {mOBiasHost.data(),   mOBiasHost.size()   * sizeof(float)};
    s[W_PERM_Q]   = {mPermQHost.data(),   mPermQHost.size()   * sizeof(int32_t)};
    s[W_ROT_Q]    = {mRotQHost.data(),    mRotQHost.size()    * sizeof(uint16_t)};
    s[W_PERM_O]   = {mPermOHost.data(),   mPermOHost.size()   * sizeof(int32_t)};
    s[W_ROT_O]    = {mRotOHost.data(),    mRotOHost.size()    * sizeof(uint16_t)};
    s[W_SCALE_PRE_IN] = {mScalePreInHost.data(), mScalePreInHost.size() * sizeof(float)};
    s[W_SCALE_PRE_O]  = {mScalePreOHost.data(),  mScalePreOHost.size()  * sizeof(float)};
    return s;
}

void FusedCrossAttnFullInt4Plugin::bindDeviceWeights() {
    mQI4Device     = mShared->buf(W_Q_I4);
    mQScaleDevice  = mShared->buf(W_Q_SCALE);
    mQBiasDevice   = mShared->buf(W_Q_BIAS);
    mKVI4Device    = mShared->buf(W_KV_I4);
    mKVScaleDevice = mShared->buf(W_KV_SCALE);
    mKVBiasDevice  = mShared->buf(W_KV_BIAS);
    mOI4Device     = mShared->buf(W_O_I4);
    mOScaleDevice  = mShared->buf(W_O_SCALE);
    mOBiasDevice   = mShared->buf(W_O_BIAS);
    mPermQDevice   = mShared->buf(W_PERM_Q);
    mRotQBf16Device = mShared->buf(W_ROT_Q);
    mPermODevice   = mShared->buf(W_PERM_O);
    mRotOBf16Device = mShared->buf(W_ROT_O);
    mScalePreInDevice = mScalePreInHost.empty() ? nullptr : mShared->buf(W_SCALE_PRE_IN);
    mScalePreODevice  = mScalePreOHost.empty()  ? nullptr : mShared->buf(W_SCALE_PRE_O);
}

void FusedCrossAttnFullInt4Plugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedCrossAttnFullInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedCrossAttnFullInt4Plugin::clone() noexcept {
    try {
        auto* p = new FusedCrossAttnFullInt4Plugin(
            mLayerName,
            mQI4Host, mQScaleHost, mQBiasHost,
            mKVI4Host, mKVScaleHost, mKVBiasHost,
            mOI4Host, mOScaleHost, mOBiasHost,
            mPermQHost, mRotQHost, mPermOHost, mRotOHost,
            mInnerDim, mK, mKEnc, mNumHeads, mHeadDim, mBlockSize, mEps);
        p->mHasMask = mHasMask;
        p->mEmitKV = mEmitKV;
        // attachToContext() clones; a member missing here is gone at runtime.
        p->mRotBlockSize = mRotBlockSize;
        p->mScalePreInHost = mScalePreInHost;
        p->mScalePreOHost = mScalePreOHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedCrossAttnFullInt4Plugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullInt4Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedCrossAttnFullInt4Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedCrossAttnFullInt4Plugin::getNbOutputs() const noexcept { return 1 + (mEmitKV ? 1 : 0); }

int32_t FusedCrossAttnFullInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1 || nbOutputs == 2);
    outputTypes[0] = DataType::kBF16;
    if (nbOutputs == 2) outputTypes[1] = DataType::kBF16;
    return 0;
}

int32_t FusedCrossAttnFullInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t /*nbInputs*/,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept {
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
    if (nbOutputs == 2) {
        auto const& enc = inputs[3];
        outputs[1].nbDims = enc.nbDims;
        for (int32_t i = 0; i < enc.nbDims - 1; ++i) outputs[1].d[i] = enc.d[i];
        outputs[1].d[enc.nbDims - 1] = exprBuilder.constant(2 * mInnerDim);
    }
    return 0;
}

bool FusedCrossAttnFullInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert((nbInputs == 5 || nbInputs == 6) && (nbOutputs == 1 || nbOutputs == 2));
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    if (pos == 0 || pos == 1 || pos == 2) return d.type == DataType::kBF16;
    if (pos == 3) return d.type == DataType::kINT32;   // enc_i4 INT32-packed
    if (pos == 4) return d.type == DataType::kFLOAT;
    if (pos == 5 && nbInputs == 6) return d.type == DataType::kBF16;
    if (pos == nbInputs) return d.type == DataType::kBF16;
    if (pos == nbInputs + 1) return d.type == DataType::kBF16;
    return false;
}

int32_t FusedCrossAttnFullInt4Plugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    mHasMask = (nbInputs == 6);
    return 0;
}

size_t FusedCrossAttnFullInt4Plugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t S = inputs[0].max.d[inputs[0].max.nbDims - 2];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    int64_t MEncMax = 1;
    for (int32_t i = 0; i < inputs[3].max.nbDims - 1; ++i) MEncMax *= inputs[3].max.d[i];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax, static_cast<int32_t>(MEncMax),
                          mInnerDim, mNumHeads, S);
}

int32_t FusedCrossAttnFullInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        if (mShared == nullptr) return -1;  // weights bound in attachToContext
        ensureCublasHandle();
        auto handle = reinterpret_cast<cublasHandle_t>(mCublasHandle);
        cublasSetStream(handle, stream);

        auto const& xDesc   = inputDesc[0];
        auto const& encDesc = inputDesc[3];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        const int32_t B = gr00t::v1::plugins::sdpa_sample_count(M, S);
        if (B < 0) return -1;  // rows do not tile S: malformed (..., S, K) input
        int32_t K = xDesc.dims.d[nbDims - 1];
        int64_t MEnc = 1;
        for (int32_t i = 0; i < encDesc.dims.nbDims - 1; ++i) MEnc *= encDesc.dims.d[i];
        const int32_t KEnc = mKEnc;
        const int32_t H = mNumHeads;
        const int32_t D = mHeadDim;

        size_t Wm = static_cast<size_t>(M) * (K > mInnerDim ? K : mInnerDim);
        size_t xp_sz = alignUp(Wm * sizeof(uint16_t));
        size_t xr_sz = alignUp(Wm * sizeof(uint16_t));
        size_t x_i4_sz   = alignUp(static_cast<size_t>(M) * K / 2);
        size_t x_sc_sz   = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t q_bf16_sz = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t kv_bf16_sz= alignUp(static_cast<size_t>(MEnc) * 2 * mInnerDim * sizeof(uint16_t));
        // MEnc is already the product of the encoder's leading dims (B*Senc), so the
        // batch factor is folded in. Multiplying by B again over-counts by B and
        // pushes every buffer after scores past the end of the granted workspace
        // (getWorkspaceSize sizes this region as numHeads*S*MEnc).
        size_t scores_sz = alignUp(static_cast<size_t>(H) * S * MEnc * sizeof(uint16_t));
        size_t attn_b_sz = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t attn_i4_sz= alignUp(static_cast<size_t>(M) * mInnerDim / 2);

        auto* ws = static_cast<uint8_t*>(workspace);
        void*   xp      = static_cast<void*>(ws);
        void*   xr      = static_cast<void*>(ws + xp_sz);
        uint8_t* rest   = ws + xp_sz + xr_sz;
        int8_t* x_i4    = reinterpret_cast<int8_t*>(rest);
        float*  x_sc    = reinterpret_cast<float*>(rest + x_i4_sz);
        void*   q_bf16  = static_cast<void*>(rest + x_i4_sz + x_sc_sz);
        void*   kv_bf16 = (mEmitKV ? outputs[1]
                                   : static_cast<void*>(static_cast<uint8_t*>(q_bf16) + q_bf16_sz));
        void*   kv_skip = static_cast<void*>(static_cast<uint8_t*>(q_bf16) + q_bf16_sz);
        void*   scores  = static_cast<void*>(static_cast<uint8_t*>(kv_skip) + kv_bf16_sz);
        void*   attn_b  = static_cast<void*>(static_cast<uint8_t*>(scores) + scores_sz);
        int8_t* attn_i4 = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(attn_b) + attn_b_sz);
        float*  attn_sc = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(attn_i4) + attn_i4_sz);

        // Step 1: AdaLN+permute(q) → cuBLAS block-rotate → per-row INT4 quant on x.
        int rc = dit_adaln_permute_bf16(
            inputs[0], inputs[1], inputs[2], mPermQDevice, xp, B, S, K, mEps, stream);
        if (rc != 0) return rc;
        if (mRotBlockSize > 0) {
            // Butterfly: the cuBLAS block GEMM and its baked matrix are gone;
            // the Hadamard is recomputed inside the quant kernel and the
            // fold-before SmoothQuant vector divides the raw channel going in.
            rc = dit_int4_per_row_quant_fwht_bf16(xp, mScalePreInDevice, nullptr, x_i4, x_sc,
                                                  static_cast<int32_t>(M), K, mRotBlockSize,
                                                  /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        } else {
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xp, mRotQBf16Device, xr,
                                            static_cast<int32_t>(M), K, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xr, x_i4, x_sc, static_cast<int32_t>(M), K, /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        }

        // Step 2: INT4 Q GEMM (M, inner) + bias.
        rc = dit_int4_rowwise_gemm_bias_bf16out(
            x_i4, mQI4Device, x_sc, mQScaleDevice, mQBiasDevice,
            q_bf16, static_cast<int32_t>(M), mInnerDim, K, stream);
        if (rc != 0) return rc;

        // Step 3: INT4 merged-KV GEMM on the pre-rotated/int4 encoder (inputs[3/4]).
        const int32_t Nkv = 2 * mInnerDim;
        rc = dit_int4_rowwise_gemm_bias_bf16out(
            inputs[3], mKVI4Device, inputs[4], mKVScaleDevice, mKVBiasDevice,
            kv_bf16, static_cast<int32_t>(MEnc), Nkv, KEnc, stream);
        if (rc != 0) return rc;

        // Steps 4-6: SDPA with optional additive mask (the same helper the INT8 path uses).
        // q_bf16 is (B, S, H*D); kv_bf16 is (B, S_enc, 2*H*D) with K at offset 0 and V
        // at offset H*D; attn_b is (B, S, H*D).
        if (B <= 0 || MEnc <= 0 || MEnc % B != 0) return -1;
        const int32_t SEnc = static_cast<int32_t>(MEnc / B);

        using gr00t::v1::plugins::SdpaOperand;
        auto* qBF16 = reinterpret_cast<__nv_bfloat16*>(q_bf16);
        auto* kvBF16 = reinterpret_cast<__nv_bfloat16*>(kv_bf16);
        const int ld_q = mInnerDim;
        const int ld_kv = 2 * mInnerDim;
        const long long q_sample = static_cast<long long>(S) * ld_q;
        const long long kv_sample = static_cast<long long>(SEnc) * ld_kv;
        const long long attn_sample = static_cast<long long>(S) * mInnerDim;

        // The mask is (B, 1, 1, S_enc): one additive row per sample.
        void const* mask_ptr = mHasMask ? inputs[5] : nullptr;
        rc = gr00t::v1::plugins::sdpa_bf16_cublas(
            handle,
            SdpaOperand{qBF16, ld_q, q_sample},
            SdpaOperand{kvBF16 + 0 * mInnerDim, ld_kv, kv_sample},
            SdpaOperand{kvBF16 + 1 * mInnerDim, ld_kv, kv_sample},
            reinterpret_cast<__nv_bfloat16*>(scores),
            reinterpret_cast<__nv_bfloat16*>(attn_b), mInnerDim, attn_sample,
            mask_ptr, /*mask_rows=*/B,
            B, H, S, SEnc, D, stream);
        if (rc != 0) return rc;

        // Step 7: permute(o) → cuBLAS block-rotate → per-row INT4 quant on attn_buf.
        rc = dit_permute_bf16(attn_b, mPermODevice, xp, static_cast<int32_t>(M), mInnerDim, stream);
        if (rc != 0) return rc;
        if (mRotBlockSize > 0) {
            // Butterfly: the cuBLAS block GEMM and its baked matrix are gone;
            // the Hadamard is recomputed inside the quant kernel and the
            // fold-before SmoothQuant vector divides the raw channel going in.
            rc = dit_int4_per_row_quant_fwht_bf16(xp, mScalePreODevice, nullptr, attn_i4, attn_sc,
                                                  static_cast<int32_t>(M), mInnerDim, mRotBlockSize,
                                                  /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        } else {
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xp, mRotOBf16Device, xr,
                                            static_cast<int32_t>(M), mInnerDim, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xr, attn_i4, attn_sc,
                                             static_cast<int32_t>(M), mInnerDim, /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        }

        // Step 8: INT4 attn_O GEMM + bias + residual=x → outputs[0].
        rc = dit_int4_rowwise_gemm_bias_residual_bf16out(
            attn_i4, mOI4Device, attn_sc, mOScaleDevice, mOBiasDevice,
            inputs[0], outputs[0], static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedCrossAttnFullInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t nbInputs,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    // Runtime-phase re-derivation (mirrors the INT8 plugin): configurePlugin only
    // runs at BUILD time, so a deserialized engine would otherwise keep the
    // constructor default mHasMask=false and silently run UNMASKED cross-attn.
    mHasMask = (nbInputs == 6);
    return 0;
}

IPluginV3* FusedCrossAttnFullInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<FusedCrossAttnFullInt4Plugin*>(clone());
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

PluginFieldCollection const* FusedCrossAttnFullInt4Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* n, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kINT8, static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* n, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(v.size())));
    };
    auto pushI32 = [&](const char* n, std::vector<int32_t>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kINT32, static_cast<int32_t>(v.size())));
    };
    auto pushBf16 = [&](const char* n, std::vector<uint16_t>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kBF16, static_cast<int32_t>(v.size())));
    };
    pushI8("weight_q_i4",    mQI4Host);   pushF32("weight_q_scale",  mQScaleHost);  pushF32("bias_q",  mQBiasHost);
    pushI8("weight_kv_i4",   mKVI4Host);  pushF32("weight_kv_scale", mKVScaleHost); pushF32("bias_kv", mKVBiasHost);
    pushI8("weight_o_i4",    mOI4Host);   pushF32("weight_o_scale",  mOScaleHost);  pushF32("bias_o",  mOBiasHost);
    pushI32("perm_q", mPermQHost);  pushBf16("rotation_q", mRotQHost);
    pushI32("perm_o", mPermOHost);  pushBf16("rotation_o", mRotOHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K_enc",     &mKEnc,     PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("num_heads", &mNumHeads, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("head_dim",  &mHeadDim,  PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("block_size",&mBlockSize,PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",       &mEps,      PluginFieldType::kFLOAT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre_in", mScalePreInHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreInHost.size())));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre_o", mScalePreOHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreOHost.size())));
    mDataToSerialize.emplace_back(PluginField("emit_kv",   &mEmitKV,   PluginFieldType::kINT32, 1));
    mHasMaskSerialized = mHasMask ? 1 : 0;
    mDataToSerialize.emplace_back(PluginField("has_mask",  &mHasMaskSerialized, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedCrossAttnFullInt4PluginCreator::FusedCrossAttnFullInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_q_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_q_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_q", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_kv", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("perm_q", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation_q", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("perm_o", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation_o", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K_enc", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_in", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("emit_kv", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedCrossAttnFullInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedCrossAttnFullInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedCrossAttnFullInt4PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedCrossAttnFullInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedCrossAttnFullInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedCrossAttnFullInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedCrossAttnFullInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedCrossAttnFullInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedCrossAttnFullInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
