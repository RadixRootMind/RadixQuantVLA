// FusedSelfAttnFullInt4: INT4 merged-QKV + cuBLAS BF16 SDPA + INT4 attn_O, with
// FoldQuant rotation folded into the QKV/O prologues. Clone of fused_selfattn_full_plugin.cpp.

#include "plugin_field_util.h"
#include "fused_selfattn_full_int4_plugin.h"
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
constexpr char const* kPLUGIN_NAME{"FusedSelfAttnFullInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace:
//   [x_i4 (M*K/2)][x_scale (M*4)][merged_qkv (M*3*inner*2)][scores (B*H*S*S*2)]
//   [attn_buf (M*inner*2)][attn_i4 (M*inner/2)][attn_scale (M*4)]
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t innerDim, int32_t numHeads, int32_t S) {
    size_t Wm = static_cast<size_t>(M) * (K > innerDim ? K : innerDim);
    size_t xp = alignUp(Wm * sizeof(uint16_t));   // pre-rotation BF16 (permuted)
    size_t xr = alignUp(Wm * sizeof(uint16_t));   // post-rotation BF16
    size_t x_i4    = alignUp(static_cast<size_t>(M) * K / 2);
    size_t x_sc    = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t merged  = alignUp(static_cast<size_t>(M) * 3 * static_cast<size_t>(innerDim) * sizeof(uint16_t));
    // scores is laid out B*H*S*S in enqueue(); M == B*S, so B == M/S. Omitting the
    // batch factor here under-allocates and every buffer placed after scores
    // overruns the TensorRT workspace.
    size_t B       = (S > 0) ? static_cast<size_t>(M) / static_cast<size_t>(S) : 1;
    size_t scores  = alignUp(B * static_cast<size_t>(numHeads) * static_cast<size_t>(S) * S * sizeof(uint16_t));
    size_t attn_b  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t attn_i4 = alignUp(static_cast<size_t>(M) * innerDim / 2);
    size_t attn_sc = alignUp(static_cast<size_t>(M) * sizeof(float));
    return xp + xr + x_i4 + x_sc + merged + scores + attn_b + attn_i4 + attn_sc;
}
}  // anon

PluginFieldCollection FusedSelfAttnFullInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> FusedSelfAttnFullInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedSelfAttnFullInt4PluginCreator);

FusedSelfAttnFullInt4Plugin::FusedSelfAttnFullInt4Plugin(std::string const& name,
    std::vector<int8_t> wQKV, std::vector<float> wQKVScale, std::vector<float> bQKV,
    std::vector<int8_t> wO,   std::vector<float> wOScale,   std::vector<float> bO,
    std::vector<int32_t> permQKV, std::vector<uint16_t> rotQKV,
    std::vector<int32_t> permO,   std::vector<uint16_t> rotO,
    int32_t innerDim, int32_t K, int32_t numHeads, int32_t headDim, int32_t blockSize, float eps)
    : mLayerName(name)
    , mQKVI4Host(std::move(wQKV)), mQKVScaleHost(std::move(wQKVScale)), mQKVBiasHost(std::move(bQKV))
    , mOI4Host(std::move(wO)), mOScaleHost(std::move(wOScale)), mOBiasHost(std::move(bO))
    , mPermQKVHost(std::move(permQKV)), mRotQKVHost(std::move(rotQKV))
    , mPermOHost(std::move(permO)), mRotOHost(std::move(rotO))
    , mInnerDim(innerDim), mK(K), mNumHeads(numHeads), mHeadDim(headDim)
    , mBlockSize(blockSize), mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedSelfAttnFullInt4Plugin::FusedSelfAttnFullInt4Plugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_qkv_i4")        { auto* p = static_cast<int8_t const*>(f.data); mQKVI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_qkv_scale"){ auto* p = static_cast<float const*>(f.data); mQKVScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_qkv")        { auto* p = static_cast<float const*>(f.data); mQKVBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_o_i4")     { auto* p = static_cast<int8_t const*>(f.data); mOI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_o_scale")  { auto* p = static_cast<float const*>(f.data); mOScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_o")          { auto* p = static_cast<float const*>(f.data); mOBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "perm_qkv")        { auto* p = static_cast<int32_t const*>(f.data); mPermQKVHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation_qkv")    { auto* p = static_cast<uint16_t const*>(f.data); mRotQKVHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "perm_o")          { auto* p = static_cast<int32_t const*>(f.data); mPermOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation_o")      { auto* p = static_cast<uint16_t const*>(f.data); mRotOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "inner_dim")       { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")               { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "num_heads")       { mNumHeads = *static_cast<int32_t const*>(f.data); }
        else if (n == "head_dim")        { mHeadDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "block_size")      { mBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "rot_block_size")   { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre_in") { auto* p = static_cast<float const*>(f.data); mScalePreInHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre_o")  { auto* p = static_cast<float const*>(f.data); mScalePreOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "eps")             { mEps = *static_cast<float const*>(f.data); }
    }
}

FusedSelfAttnFullInt4Plugin::~FusedSelfAttnFullInt4Plugin() {
    // The mXDevice pointers are non-owning views into the shared resource; only the
    // per-context cuBLAS handle is owned here. Release our ref on the shared weights.
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights(). Rotation
// buffers are BF16 raw (uint16), everything else is byte-copied verbatim.
enum SelfAttnW {
    W_QKV_I4 = 0, W_QKV_SCALE, W_QKV_BIAS,
    W_O_I4, W_O_SCALE, W_O_BIAS,
    W_PERM_QKV, W_ROT_QKV, W_PERM_O, W_ROT_O, W_SCALE_PRE_IN, W_SCALE_PRE_O, W_COUNT
};
}  // anon

std::vector<WeightSpec> FusedSelfAttnFullInt4Plugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_QKV_I4]    = {mQKVI4Host.data(),    mQKVI4Host.size()};
    s[W_QKV_SCALE] = {mQKVScaleHost.data(), mQKVScaleHost.size() * sizeof(float)};
    s[W_QKV_BIAS]  = {mQKVBiasHost.data(),  mQKVBiasHost.size()  * sizeof(float)};
    s[W_O_I4]      = {mOI4Host.data(),      mOI4Host.size()};
    s[W_O_SCALE]   = {mOScaleHost.data(),   mOScaleHost.size()   * sizeof(float)};
    s[W_O_BIAS]    = {mOBiasHost.data(),    mOBiasHost.size()    * sizeof(float)};
    s[W_PERM_QKV]  = {mPermQKVHost.data(),  mPermQKVHost.size()  * sizeof(int32_t)};
    s[W_ROT_QKV]   = {mRotQKVHost.data(),   mRotQKVHost.size()   * sizeof(uint16_t)};
    s[W_PERM_O]    = {mPermOHost.data(),    mPermOHost.size()    * sizeof(int32_t)};
    s[W_ROT_O]     = {mRotOHost.data(),     mRotOHost.size()     * sizeof(uint16_t)};
    s[W_SCALE_PRE_IN] = {mScalePreInHost.data(), mScalePreInHost.size() * sizeof(float)};
    s[W_SCALE_PRE_O]  = {mScalePreOHost.data(),  mScalePreOHost.size()  * sizeof(float)};
    return s;
}

void FusedSelfAttnFullInt4Plugin::bindDeviceWeights() {
    mQKVI4Device      = mShared->buf(W_QKV_I4);
    mQKVScaleDevice   = mShared->buf(W_QKV_SCALE);
    mQKVBiasDevice    = mShared->buf(W_QKV_BIAS);
    mOI4Device        = mShared->buf(W_O_I4);
    mOScaleDevice     = mShared->buf(W_O_SCALE);
    mOBiasDevice      = mShared->buf(W_O_BIAS);
    mPermQKVDevice    = mShared->buf(W_PERM_QKV);
    mRotQKVBf16Device = mShared->buf(W_ROT_QKV);
    mPermODevice      = mShared->buf(W_PERM_O);
    mRotOBf16Device   = mShared->buf(W_ROT_O);
    mScalePreInDevice = mScalePreInHost.empty() ? nullptr : mShared->buf(W_SCALE_PRE_IN);
    mScalePreODevice  = mScalePreOHost.empty()  ? nullptr : mShared->buf(W_SCALE_PRE_O);
}

void FusedSelfAttnFullInt4Plugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedSelfAttnFullInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedSelfAttnFullInt4Plugin::clone() noexcept {
    try {
        auto* p = new FusedSelfAttnFullInt4Plugin(
            mLayerName,
            mQKVI4Host, mQKVScaleHost, mQKVBiasHost,
            mOI4Host, mOScaleHost, mOBiasHost,
            mPermQKVHost, mRotQKVHost, mPermOHost, mRotOHost,
            mInnerDim, mK, mNumHeads, mHeadDim, mBlockSize, mEps);
        // attachToContext() clones; a member missing here is gone at runtime.
        p->mRotBlockSize = mRotBlockSize;
        p->mScalePreInHost = mScalePreInHost;
        p->mScalePreOHost = mScalePreOHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedSelfAttnFullInt4Plugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedSelfAttnFullInt4Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedSelfAttnFullInt4Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedSelfAttnFullInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedSelfAttnFullInt4Plugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedSelfAttnFullInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedSelfAttnFullInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /*exprBuilder*/) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
    return 0;
}

bool FusedSelfAttnFullInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedSelfAttnFullInt4Plugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedSelfAttnFullInt4Plugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t S = inputs[0].max.d[inputs[0].max.nbDims - 2];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax, mInnerDim, mNumHeads, S);
}

int32_t FusedSelfAttnFullInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        // Weights already reside in the shared device resource (bound in
        // attachToContext); only the per-context cuBLAS handle is lazily created.
        if (mShared == nullptr) return -1;
        ensureCublasHandle();
        auto handle = reinterpret_cast<cublasHandle_t>(mCublasHandle);
        cublasSetStream(handle, stream);

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        const int32_t B = gr00t::v1::plugins::sdpa_sample_count(M, S);
        if (B < 0) return -1;  // rows do not tile S: malformed (..., S, K) input
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);
        const int32_t H = mNumHeads;
        const int32_t D = mHeadDim;

        size_t Wm = static_cast<size_t>(M) * (K > mInnerDim ? K : mInnerDim);
        size_t xp_sz = alignUp(Wm * sizeof(uint16_t));
        size_t xr_sz = alignUp(Wm * sizeof(uint16_t));
        size_t x_i4_sz    = alignUp(static_cast<size_t>(M) * K / 2);
        size_t x_sc_sz    = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t merged_sz  = alignUp(static_cast<size_t>(M) * 3 * static_cast<size_t>(mInnerDim) * sizeof(uint16_t));
        size_t scores_sz  = alignUp(static_cast<size_t>(B) * H * S * S * sizeof(uint16_t));
        size_t attn_b_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t attn_i4_sz = alignUp(static_cast<size_t>(M) * mInnerDim / 2);

        auto* ws = static_cast<uint8_t*>(workspace);
        void*   xp      = static_cast<void*>(ws);
        void*   xr      = static_cast<void*>(ws + xp_sz);
        uint8_t* rest   = ws + xp_sz + xr_sz;
        int8_t* x_i4    = reinterpret_cast<int8_t*>(rest);
        float*  x_sc    = reinterpret_cast<float*>(rest + x_i4_sz);
        void*   merged  = static_cast<void*>(rest + x_i4_sz + x_sc_sz);
        void*   scores  = static_cast<void*>(static_cast<uint8_t*>(merged) + merged_sz);
        void*   attn_b  = static_cast<void*>(static_cast<uint8_t*>(scores) + scores_sz);
        int8_t* attn_i4 = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(attn_b) + attn_b_sz);
        float*  attn_sc = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(attn_i4) + attn_i4_sz);

        // Step 1: AdaLN+permute(qkv) → cuBLAS block-rotate → per-row INT4 quant on x.
        int rc = dit_adaln_permute_bf16(
            inputs[0], inputs[1], inputs[2], mPermQKVDevice, xp, B, S, K, mEps, stream);
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
            rc = dit_int4_block_rotate_bf16(mCublasHandle, xp, mRotQKVBf16Device, xr,
                                            static_cast<int32_t>(M), K, mBlockSize, stream);
            if (rc != 0) return rc;
            rc = dit_int4_per_row_quant_bf16(xr, x_i4, x_sc, static_cast<int32_t>(M), K, /*act_clip=*/1.0f, stream);
            if (rc != 0) return rc;
        }

        // Step 2: INT4 merged-QKV GEMM + bias → BF16 (M, 3*inner), layout (S,3,H,D).
        int32_t Ntot = 3 * mInnerDim;
        rc = dit_int4_rowwise_gemm_bias_bf16out(
            x_i4, mQKVI4Device, x_sc, mQKVScaleDevice, mQKVBiasDevice,
            merged, static_cast<int32_t>(M), Ntot, K, stream);
        if (rc != 0) return rc;

        // Steps 3-5: BF16 cuBLAS SDPA (the same helper the INT8 path uses).
        // `merged` is (B, S, 3, H, D) row-major; `attn_b` is (B, S, H*D).
        using gr00t::v1::plugins::SdpaOperand;
        auto* mergedBF16 = reinterpret_cast<__nv_bfloat16*>(merged);
        const int ld_qkv = 3 * mInnerDim;
        const long long qkv_sample = static_cast<long long>(S) * ld_qkv;
        const long long attn_sample = static_cast<long long>(S) * mInnerDim;

        rc = gr00t::v1::plugins::sdpa_bf16_cublas(
            handle,
            SdpaOperand{mergedBF16 + 0 * mInnerDim, ld_qkv, qkv_sample},
            SdpaOperand{mergedBF16 + 1 * mInnerDim, ld_qkv, qkv_sample},
            SdpaOperand{mergedBF16 + 2 * mInnerDim, ld_qkv, qkv_sample},
            reinterpret_cast<__nv_bfloat16*>(scores),
            reinterpret_cast<__nv_bfloat16*>(attn_b), mInnerDim, attn_sample,
            /*mask_bf16=*/nullptr, /*mask_rows=*/0,
            B, H, S, S, D, stream);
        if (rc != 0) return rc;

        // Step 6: permute(o) → cuBLAS block-rotate → per-row INT4 quant on attn_buf.
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

        // Step 7: INT4 attn_O GEMM + bias + residual=x → outputs[0].
        rc = dit_int4_rowwise_gemm_bias_residual_bf16out(
            attn_i4, mOI4Device, attn_sc, mOScaleDevice, mOBiasDevice,
            inputs[0], outputs[0], static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedSelfAttnFullInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedSelfAttnFullInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<FusedSelfAttnFullInt4Plugin*>(clone());
        if (p == nullptr) return nullptr;
        // Acquire (or create) one shared device copy of the immutable weights,
        // keyed by content digest → every execution context shares it.
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

PluginFieldCollection const* FusedSelfAttnFullInt4Plugin::getFieldsToSerialize() noexcept {
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
    pushI8("weight_qkv_i4",    mQKVI4Host);
    pushF32("weight_qkv_scale", mQKVScaleHost);
    pushF32("bias_qkv",         mQKVBiasHost);
    pushI8("weight_o_i4",      mOI4Host);
    pushF32("weight_o_scale",   mOScaleHost);
    pushF32("bias_o",           mOBiasHost);
    pushI32("perm_qkv", mPermQKVHost);
    pushBf16("rotation_qkv", mRotQKVHost);
    pushI32("perm_o", mPermOHost);
    pushBf16("rotation_o", mRotOHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("num_heads", &mNumHeads, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("head_dim",  &mHeadDim,  PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("block_size", &mBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",       &mEps,      PluginFieldType::kFLOAT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre_in", mScalePreInHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreInHost.size())));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre_o", mScalePreOHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreOHost.size())));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedSelfAttnFullInt4PluginCreator::FusedSelfAttnFullInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_qkv_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_qkv_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_qkv", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("perm_qkv", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation_qkv", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("perm_o", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation_o", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_in", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedSelfAttnFullInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedSelfAttnFullInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedSelfAttnFullInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedSelfAttnFullInt4PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedSelfAttnFullInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedSelfAttnFullInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedSelfAttnFullInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedSelfAttnFullInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedSelfAttnFullInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedSelfAttnFullInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedSelfAttnFullInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
