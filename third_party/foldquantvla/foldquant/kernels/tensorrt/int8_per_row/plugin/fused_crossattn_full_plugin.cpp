// FusedCrossAttnFull: AdaLN + Q INT8 GEMM + KV INT8 GEMM (from pre-quantized
// encoder) + cuBLAS BF16 SDPA (optionally masked) + INT8 attn_O + bias +
// residual collapsed into one plugin call.

#include "plugin_field_util.h"
#include "fused_crossattn_full_plugin.h"
#include "fused_adaln_quant.h"
#include "dit_int8_rowwise_v2.h"
#include "dit_int8_rowwise_v2_fused.h"
#include "sdpa_cublas.h"

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
constexpr char const* kPLUGIN_NAME{"FusedCrossAttnFull"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace:
//   x_i8 (M*K)              | x_scale (M*4)
//   Q_bf16 (M * inner * 2)              ← INT8 GEMM output (S, inner_dim)
//   KV_bf16 (M_enc * 2*inner * 2)       ← merged K,V (S_enc, 2, H, D)
//   scores (B * H * S * M_enc * 2)      ← Q·Kᵀ result
//   attn_buf (M * inner * 2)            ← scores·V output (S, H, D) interleaved
//   attn_i8 (M * inner) | attn_scale (M*4)
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t MEnc, int32_t innerDim,
                              int32_t H, int32_t S) {
    size_t x_i8    = alignUp(static_cast<size_t>(M) * K);
    size_t x_sc    = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t q_bf16  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t kv_bf16 = alignUp(static_cast<size_t>(MEnc) * 2 * innerDim * sizeof(uint16_t));
    size_t scores  = alignUp(static_cast<size_t>(H) * S * MEnc * sizeof(uint16_t));
    size_t attn_b  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t attn_i8 = alignUp(static_cast<size_t>(M) * innerDim);
    size_t attn_sc = alignUp(static_cast<size_t>(M) * sizeof(float));
    return x_i8 + x_sc + q_bf16 + kv_bf16 + scores + attn_b + attn_i8 + attn_sc;
}
}  // anon

PluginFieldCollection FusedCrossAttnFullPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedCrossAttnFullPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedCrossAttnFullPluginCreator);

FusedCrossAttnFullPlugin::FusedCrossAttnFullPlugin(std::string const& name,
    std::vector<int8_t> wQ,  std::vector<float> wQScale,  std::vector<float> bQ,
    std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
    std::vector<int8_t> wO,  std::vector<float> wOScale,  std::vector<float> bO,
    int32_t innerDim, int32_t K, int32_t KEnc,
    int32_t numHeads, int32_t headDim, float eps)
    : mLayerName(name)
    , mQI8Host(std::move(wQ)),   mKVI8Host(std::move(wKV)),   mOI8Host(std::move(wO))
    , mQScaleHost(std::move(wQScale)), mKVScaleHost(std::move(wKVScale)), mOScaleHost(std::move(wOScale))
    , mQBiasHost(std::move(bQ)),  mKVBiasHost(std::move(bKV)),  mOBiasHost(std::move(bO))
    , mInnerDim(innerDim), mK(K), mKEnc(KEnc)
    , mNumHeads(numHeads), mHeadDim(headDim), mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedCrossAttnFullPlugin::FusedCrossAttnFullPlugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if      (n == "weight_q_i8")     { auto* p = static_cast<int8_t const*>(f.data); mQI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_q_scale")  { auto* p = static_cast<float const*>(f.data);  mQScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_q")          { auto* p = static_cast<float const*>(f.data);  mQBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_kv_i8")    { auto* p = static_cast<int8_t const*>(f.data); mKVI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_kv_scale") { auto* p = static_cast<float const*>(f.data);  mKVScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_kv")         { auto* p = static_cast<float const*>(f.data);  mKVBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_o_i8")     { auto* p = static_cast<int8_t const*>(f.data); mOI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_o_scale")  { auto* p = static_cast<float const*>(f.data);  mOScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_o")          { auto* p = static_cast<float const*>(f.data);  mOBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_x")         { auto* p = static_cast<float const*>(f.data); mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_post_sdpa") { auto* p = static_cast<float const*>(f.data); mStaticActScalePostSdpaHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "inner_dim") { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")         { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "K_enc")     { mKEnc = *static_cast<int32_t const*>(f.data); }
        else if (n == "num_heads") { mNumHeads = *static_cast<int32_t const*>(f.data); }
        else if (n == "head_dim")  { mHeadDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")       { mEps = *static_cast<float const*>(f.data); }
        else if (n == "rot_block_size")  { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre_in") { auto* p = static_cast<float const*>(f.data); mScalePreInHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre_o")  { auto* p = static_cast<float const*>(f.data); mScalePreOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "has_mask")  { mHasMask = (*static_cast<int32_t const*>(f.data)) != 0; }
        else if (n == "emit_kv")   { mEmitKV = *static_cast<int32_t const*>(f.data); }
    }
}

FusedCrossAttnFullPlugin::~FusedCrossAttnFullPlugin() {
    auto free_if = [](void* p) { if (p) cudaFree(p); };
    free_if(mQI8Device);  free_if(mQScaleDevice);  free_if(mQBiasDevice);
    free_if(mKVI8Device); free_if(mKVScaleDevice); free_if(mKVBiasDevice);
    free_if(mOI8Device);  free_if(mOScaleDevice);  free_if(mOBiasDevice);
    free_if(mStaticActScaleXDevice);
    free_if(mStaticActScalePostSdpaDevice);
    free_if(mScalePreInDevice);
    free_if(mScalePreODevice);
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
}

void FusedCrossAttnFullPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mScalePreInDevice, mScalePreInHost.data(), mScalePreInHost.size() * sizeof(float));
    upload(&mScalePreODevice,  mScalePreOHost.data(),  mScalePreOHost.size()  * sizeof(float));
    upload(&mQI8Device,   mQI8Host.data(),   mQI8Host.size());
    upload(&mQScaleDevice, mQScaleHost.data(), mQScaleHost.size() * sizeof(float));
    upload(&mQBiasDevice,  mQBiasHost.data(),  mQBiasHost.size()  * sizeof(float));
    upload(&mKVI8Device,   mKVI8Host.data(),   mKVI8Host.size());
    upload(&mKVScaleDevice, mKVScaleHost.data(), mKVScaleHost.size() * sizeof(float));
    upload(&mKVBiasDevice,  mKVBiasHost.data(),  mKVBiasHost.size()  * sizeof(float));
    upload(&mOI8Device,    mOI8Host.data(),    mOI8Host.size());
    upload(&mOScaleDevice, mOScaleHost.data(), mOScaleHost.size() * sizeof(float));
    upload(&mOBiasDevice,  mOBiasHost.data(),  mOBiasHost.size()  * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
    upload(&mStaticActScalePostSdpaDevice, mStaticActScalePostSdpaHost.data(),
           mStaticActScalePostSdpaHost.size() * sizeof(float));
}

void FusedCrossAttnFullPlugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedCrossAttnFullPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedCrossAttnFullPlugin::clone() noexcept {
    try {
        auto* p = new FusedCrossAttnFullPlugin(
            mLayerName,
            mQI8Host,  mQScaleHost,  mQBiasHost,
            mKVI8Host, mKVScaleHost, mKVBiasHost,
            mOI8Host,  mOScaleHost,  mOBiasHost,
            mInnerDim, mK, mKEnc, mNumHeads, mHeadDim, mEps);
        p->mStaticActScaleXHost        = mStaticActScaleXHost;
        p->mStaticActScalePostSdpaHost = mStaticActScalePostSdpaHost;
        // The clone is what runs; a member dropped here silently stops the fold.
        p->mRotBlockSize    = mRotBlockSize;
        p->mScalePreInHost  = mScalePreInHost;
        p->mScalePreOHost   = mScalePreOHost;
        p->mHasMask                    = mHasMask;
        p->mEmitKV                     = mEmitKV;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedCrossAttnFullPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedCrossAttnFullPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedCrossAttnFullPlugin::getNbOutputs() const noexcept { return 1 + (mEmitKV ? 1 : 0); }

int32_t FusedCrossAttnFullPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1 || nbOutputs == 2);
    outputTypes[0] = DataType::kBF16;
    if (nbOutputs == 2) outputTypes[1] = DataType::kBF16;  // kv cache, BF16
    return 0;
}

int32_t FusedCrossAttnFullPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    // Input layout is either:
    //   5 inputs (legacy / no mask): x, scale, shift, enc_i8, enc_scale
    //   6 inputs (with mask):        + attn_mask (BF16 additive, (B, 1, 1, S_enc))
    assert((nbInputs == 5 || nbInputs == 6) && (nbOutputs == 1 || nbOutputs == 2));
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
    if (nbOutputs == 2) {
        // KV cache shape: (B, S_enc, 2 * inner_dim).
        // Derive from encoder INT32-packed input (idx 3): dims = (B, S_enc, K_enc / 4 of int32)
        // and from x input (idx 0): last dim = K (= inner_dim for symmetric self/cross dims),
        // but the actual inner_dim is mInnerDim. So we construct (B, S_enc, 2 * inner_dim).
        auto const& enc = inputs[3];
        outputs[1].nbDims = enc.nbDims;  // typically 3: (B, S_enc, K_enc/4)
        // Copy B and S_enc from enc, set last dim to 2 * inner_dim using a constant.
        for (int32_t i = 0; i < enc.nbDims - 1; ++i) outputs[1].d[i] = enc.d[i];
        outputs[1].d[enc.nbDims - 1] = exprBuilder.constant(2 * mInnerDim);
    }
    return 0;
}

bool FusedCrossAttnFullPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert((nbInputs == 5 || nbInputs == 6) && (nbOutputs == 1 || nbOutputs == 2));
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    if (pos == 0 || pos == 1 || pos == 2) return d.type == DataType::kBF16;  // x, scale, shift
    if (pos == 3) return d.type == DataType::kINT32;  // enc_i8 INT32-packed
    if (pos == 4) return d.type == DataType::kFLOAT;  // enc_scale
    if (pos == 5 && nbInputs == 6) return d.type == DataType::kBF16;  // attn_mask
    // Outputs are after all inputs.
    if (pos == nbInputs) return d.type == DataType::kBF16;          // post_attn
    if (pos == nbInputs + 1) return d.type == DataType::kBF16;      // kv_bf16 cache
    return false;
}

int32_t FusedCrossAttnFullPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    mHasMask = (nbInputs == 6);
    return 0;
}

size_t FusedCrossAttnFullPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t S = inputs[0].max.d[inputs[0].max.nbDims - 2];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    int64_t MEncMax = 1;
    for (int32_t i = 0; i < inputs[3].max.nbDims - 1; ++i) MEncMax *= inputs[3].max.d[i];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax,
                          static_cast<int32_t>(MEncMax), mInnerDim, mNumHeads, S);
}

int32_t FusedCrossAttnFullPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();
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

        // Workspace partition
        size_t x_i8_sz    = alignUp(static_cast<size_t>(M) * K);
        size_t x_sc_sz    = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t q_bf16_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t kv_bf16_sz = alignUp(static_cast<size_t>(MEnc) * 2 * mInnerDim * sizeof(uint16_t));
        // MEnc already folds in the batch factor (product of the encoder's leading
        // dims), so multiplying by B over-counts and pushes every buffer after
        // scores past the workspace getWorkspaceSize granted (numHeads*S*MEnc).
        size_t scores_sz  = alignUp(static_cast<size_t>(H) * S * MEnc * sizeof(uint16_t));
        size_t attn_b_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t attn_i8_sz = alignUp(static_cast<size_t>(M) * mInnerDim);

        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* x_i8     = reinterpret_cast<int8_t*>(ws);
        float*  x_sc     = reinterpret_cast<float*>(ws + x_i8_sz);
        void*   q_bf16   = static_cast<void*>(ws + x_i8_sz + x_sc_sz);
        // KV destination: if emit_kv, write directly to outputs[1] (graph-visible cache);
        // otherwise use the next workspace slice.
        void*   kv_bf16  = (mEmitKV ? outputs[1]
                                    : static_cast<void*>(static_cast<uint8_t*>(q_bf16) + q_bf16_sz));
        // The downstream scores/attn buffers still need to come from workspace;
        // skip the KV slice in workspace if we're emitting to outputs[1] (same size,
        // but cleaner to keep the layout fixed).
        void*   kv_ws_skip = static_cast<void*>(static_cast<uint8_t*>(q_bf16) + q_bf16_sz);
        void*   scores   = static_cast<void*>(static_cast<uint8_t*>(kv_ws_skip) + kv_bf16_sz);
        void*   attn_b   = static_cast<void*>(static_cast<uint8_t*>(scores) + scores_sz);
        int8_t* attn_i8  = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(attn_b) + attn_b_sz);
        float*  attn_sc  = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(attn_i8) + attn_i8_sz);

        // --- Step 1: AdaLN + per-row INT8 quant on x ---
        int rc;
        if (mRotBlockSize > 0) {
            // FoldQuant: adaLN, raw-frame divide, butterfly, then the per-row
            // amax, taken on the rotated row, which is what INT8 stores.
            rc = fused_adaln_fwht_quant_bf16_to_int8(
                inputs[0], inputs[1], inputs[2], mScalePreInDevice,
                x_i8, x_sc,
                B, S, K, mEps, mRotBlockSize, stream);
        } else if (mStaticActScaleXDevice != nullptr &&
            (int)mStaticActScaleXHost.size() >= static_cast<int>(M)) {
            rc = fused_adaln_static_quant_bf16_to_int8(
                inputs[0], inputs[1], inputs[2],
                mStaticActScaleXDevice,
                x_i8, x_sc,
                B, S, K, mEps, stream);
        } else {
            rc = fused_adaln_quant_bf16_to_int8(
                inputs[0], inputs[1], inputs[2],
                x_i8, x_sc,
                B, S, K, mEps, stream);
        }
        if (rc != 0) return rc;

        // --- Step 2: Q INT8 GEMM (M=S, N=inner_dim, K=K) ---
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            x_i8, mQI8Device,
            x_sc, mQScaleDevice, mQBiasDevice,
            q_bf16,
            static_cast<int32_t>(M), mInnerDim, K, stream);
        if (rc != 0) return rc;

        // --- Step 3: KV merged INT8 GEMM on pre-quantized encoder ---
        // Encoder INT8 = inputs[3], encoder scale = inputs[4]. Output: (M_enc, 2*inner_dim) BF16.
        const int32_t Nkv = 2 * mInnerDim;
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            inputs[3], mKVI8Device,
            inputs[4], mKVScaleDevice, mKVBiasDevice,
            kv_bf16,
            static_cast<int32_t>(MEnc), Nkv, KEnc, stream);
        if (rc != 0) return rc;

        // --- Steps 4-6: SDPA (Q·Kᵀ, masked softmax, P·V) ---
        // q_bf16 is (B, S, H*D); kv_bf16 is (B, S_enc, 2*H*D) with K at offset 0 and
        // V at offset H*D; attn_b is (B, S, H*D). MEnc counts every encoder row, so
        // the per-sample key length is MEnc / B.
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

        // --- Step 7: per-row quant on attn_buf (M, inner_dim) ---
        if (mRotBlockSize > 0) {
            rc = dit_int8_per_row_quant_fwht_bf16(
                attn_b, mScalePreODevice, /*act_scale_ch=*/nullptr,
                attn_i8, attn_sc,
                static_cast<int32_t>(M), mInnerDim, mRotBlockSize, stream);
        } else if (mStaticActScalePostSdpaDevice != nullptr &&
            (int)mStaticActScalePostSdpaHost.size() >= static_cast<int>(M)) {
            rc = dit_int8_per_row_static_quant_bf16(
                attn_b, mStaticActScalePostSdpaDevice,
                attn_i8, attn_sc,
                static_cast<int32_t>(M), mInnerDim, stream);
        } else {
            rc = dit_int8_per_row_quant_bf16(
                attn_b, attn_i8, attn_sc,
                static_cast<int32_t>(M), mInnerDim, stream);
        }
        if (rc != 0) return rc;

        // --- Step 8: INT8 attn_O GEMM + bias + residual=x → outputs[0] ---
        rc = dit_int8_rowwise_gemm_bias_residual_bf16out(
            attn_i8, mOI8Device,
            attn_sc, mOScaleDevice, mOBiasDevice,
            inputs[0],
            outputs[0],
            static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedCrossAttnFullPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t nbInputs,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    mHasMask = (nbInputs == 6);
    return 0;
}

IPluginV3* FusedCrossAttnFullPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedCrossAttnFullPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* n, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kINT8,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* n, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kFLOAT32,
                                                  static_cast<int32_t>(v.size())));
    };
    pushI8("weight_q_i8",     mQI8Host);
    pushF32("weight_q_scale",  mQScaleHost);
    pushF32("bias_q",          mQBiasHost);
    pushI8("weight_kv_i8",    mKVI8Host);
    pushF32("weight_kv_scale", mKVScaleHost);
    pushF32("bias_kv",         mKVBiasHost);
    pushI8("weight_o_i8",     mOI8Host);
    pushF32("weight_o_scale",  mOScaleHost);
    pushF32("bias_o",          mOBiasHost);
    if (!mStaticActScaleXHost.empty())        pushF32("static_act_scale_x",         mStaticActScaleXHost);
    if (!mStaticActScalePostSdpaHost.empty()) pushF32("static_act_scale_post_sdpa", mStaticActScalePostSdpaHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K_enc",     &mKEnc,     PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("num_heads", &mNumHeads, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("head_dim",  &mHeadDim,  PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",       &mEps,      PluginFieldType::kFLOAT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    if (!mScalePreInHost.empty()) {
        mDataToSerialize.emplace_back(PluginField("act_scale_pre_in", mScalePreInHost.data(),
            PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreInHost.size())));
    }
    if (!mScalePreOHost.empty()) {
        mDataToSerialize.emplace_back(PluginField("act_scale_pre_o", mScalePreOHost.data(),
            PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreOHost.size())));
    }
    mHasMaskSerialized = mHasMask ? 1 : 0;
    mDataToSerialize.emplace_back(PluginField("has_mask",  &mHasMaskSerialized, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("emit_kv",   &mEmitKV,            PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedCrossAttnFullPluginCreator::FusedCrossAttnFullPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_q_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_q_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_q", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_kv", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_post_sdpa", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K_enc", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_in", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("emit_kv",  nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedCrossAttnFullPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedCrossAttnFullPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedCrossAttnFullPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedCrossAttnFullPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedCrossAttnFullPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedCrossAttnFull(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedCrossAttnFullPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedCrossAttnFullPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedCrossAttnFullPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
