// FusedCrossAttnFullCached: cached-KV variant. KV cache provided as input
// (from FusedCrossAttnFull at step 0 with emit_kv=1). Internally:
//   AdaLN + per-row INT8 quant on x → x_i8
//   Q INT8 GEMM → q_bf16 (M, inner_dim)
//   READ K, V from inputs[3] (interleaved BF16, (M_enc, 2*inner_dim))
//   cuBLAS BF16 Q·Kᵀ → scores
//   masked softmax → scores
//   cuBLAS BF16 scores·V → attn_buf
//   per-row INT8 quant on attn_buf → attn_i8
//   attn_O INT8 GEMM + bias + residual → outputs[0]

#include "plugin_field_util.h"
#include "fused_crossattn_full_cached_plugin.h"
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
constexpr char const* kPLUGIN_NAME{"FusedCrossAttnFullCached"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace (no kv_bf16; that's provided as input):
//   x_i8 (M*K) | x_scale (M*4) | Q_bf16 (M*inner*2) | scores (B*H*S*M_enc*2)
//   | attn_buf (M*inner*2) | attn_i8 (M*inner) | attn_scale (M*4)
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t MEnc, int32_t innerDim,
                              int32_t H, int32_t S) {
    size_t x_i8    = alignUp(static_cast<size_t>(M) * K);
    size_t x_sc    = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t q_bf16  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t scores  = alignUp(static_cast<size_t>(H) * S * MEnc * sizeof(uint16_t));
    size_t attn_b  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t attn_i8 = alignUp(static_cast<size_t>(M) * innerDim);
    size_t attn_sc = alignUp(static_cast<size_t>(M) * sizeof(float));
    return x_i8 + x_sc + q_bf16 + scores + attn_b + attn_i8 + attn_sc;
}
}  // anon

PluginFieldCollection FusedCrossAttnFullCachedPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedCrossAttnFullCachedPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedCrossAttnFullCachedPluginCreator);

FusedCrossAttnFullCachedPlugin::FusedCrossAttnFullCachedPlugin(std::string const& name,
    std::vector<int8_t> wQ, std::vector<float> wQScale, std::vector<float> bQ,
    std::vector<int8_t> wO, std::vector<float> wOScale, std::vector<float> bO,
    int32_t innerDim, int32_t K,
    int32_t numHeads, int32_t headDim, float eps)
    : mLayerName(name)
    , mQI8Host(std::move(wQ)), mOI8Host(std::move(wO))
    , mQScaleHost(std::move(wQScale)), mOScaleHost(std::move(wOScale))
    , mQBiasHost(std::move(bQ)),  mOBiasHost(std::move(bO))
    , mInnerDim(innerDim), mK(K)
    , mNumHeads(numHeads), mHeadDim(headDim), mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedCrossAttnFullCachedPlugin::FusedCrossAttnFullCachedPlugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if      (n == "weight_q_i8")     { auto* p = static_cast<int8_t const*>(f.data); mQI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_q_scale")  { auto* p = static_cast<float const*>(f.data);  mQScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_q")          { auto* p = static_cast<float const*>(f.data);  mQBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_o_i8")     { auto* p = static_cast<int8_t const*>(f.data); mOI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_o_scale")  { auto* p = static_cast<float const*>(f.data);  mOScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_o")          { auto* p = static_cast<float const*>(f.data);  mOBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_x")         { auto* p = static_cast<float const*>(f.data); mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_post_sdpa") { auto* p = static_cast<float const*>(f.data); mStaticActScalePostSdpaHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "inner_dim") { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")         { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "num_heads") { mNumHeads = *static_cast<int32_t const*>(f.data); }
        else if (n == "head_dim")  { mHeadDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")       { mEps = *static_cast<float const*>(f.data); }
        else if (n == "has_mask")  { mHasMask = (*static_cast<int32_t const*>(f.data)) != 0; }
    }
}

FusedCrossAttnFullCachedPlugin::~FusedCrossAttnFullCachedPlugin() {
    auto free_if = [](void* p) { if (p) cudaFree(p); };
    free_if(mQI8Device);  free_if(mQScaleDevice);  free_if(mQBiasDevice);
    free_if(mOI8Device);  free_if(mOScaleDevice);  free_if(mOBiasDevice);
    free_if(mStaticActScaleXDevice);
    free_if(mStaticActScalePostSdpaDevice);
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
}

void FusedCrossAttnFullCachedPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mQI8Device,   mQI8Host.data(),   mQI8Host.size());
    upload(&mQScaleDevice, mQScaleHost.data(), mQScaleHost.size() * sizeof(float));
    upload(&mQBiasDevice,  mQBiasHost.data(),  mQBiasHost.size()  * sizeof(float));
    upload(&mOI8Device,    mOI8Host.data(),    mOI8Host.size());
    upload(&mOScaleDevice, mOScaleHost.data(), mOScaleHost.size() * sizeof(float));
    upload(&mOBiasDevice,  mOBiasHost.data(),  mOBiasHost.size()  * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
    upload(&mStaticActScalePostSdpaDevice, mStaticActScalePostSdpaHost.data(),
           mStaticActScalePostSdpaHost.size() * sizeof(float));
}

void FusedCrossAttnFullCachedPlugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedCrossAttnFullCachedPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedCrossAttnFullCachedPlugin::clone() noexcept {
    try {
        auto* p = new FusedCrossAttnFullCachedPlugin(
            mLayerName,
            mQI8Host, mQScaleHost, mQBiasHost,
            mOI8Host, mOScaleHost, mOBiasHost,
            mInnerDim, mK, mNumHeads, mHeadDim, mEps);
        p->mStaticActScaleXHost        = mStaticActScaleXHost;
        p->mStaticActScalePostSdpaHost = mStaticActScalePostSdpaHost;
        p->mHasMask                    = mHasMask;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedCrossAttnFullCachedPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullCachedPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedCrossAttnFullCachedPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullCachedPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedCrossAttnFullCachedPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedCrossAttnFullCachedPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedCrossAttnFullCachedPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept {
    // Input layout: 4 inputs (no mask): x, scale, shift, kv_bf16
    //               5 inputs (with mask): + attn_mask
    assert((nbInputs == 4 || nbInputs == 5) && nbOutputs == 1);
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
    return 0;
}

bool FusedCrossAttnFullCachedPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert((nbInputs == 4 || nbInputs == 5) && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    // 0..2: x, scale, shift  (BF16)
    if (pos == 0 || pos == 1 || pos == 2) return d.type == DataType::kBF16;
    // 3: kv_bf16 cached  (BF16)
    if (pos == 3) return d.type == DataType::kBF16;
    // 4: attn_mask (when present)
    if (pos == 4 && nbInputs == 5) return d.type == DataType::kBF16;
    // 5 (or 4 when no mask): output post_attn
    if (pos == nbInputs) return d.type == DataType::kBF16;
    return false;
}

int32_t FusedCrossAttnFullCachedPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    mHasMask = (nbInputs == 5);
    return 0;
}

size_t FusedCrossAttnFullCachedPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t S = inputs[0].max.d[inputs[0].max.nbDims - 2];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    // kv_bf16 input is (B, S_enc, 2*inner_dim). M_enc = product of all but last.
    int64_t MEncMax = 1;
    for (int32_t i = 0; i < inputs[3].max.nbDims - 1; ++i) MEncMax *= inputs[3].max.d[i];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax,
                          static_cast<int32_t>(MEncMax), mInnerDim, mNumHeads, S);
}

int32_t FusedCrossAttnFullCachedPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();
        ensureCublasHandle();
        auto handle = reinterpret_cast<cublasHandle_t>(mCublasHandle);
        cublasSetStream(handle, stream);

        auto const& xDesc  = inputDesc[0];
        auto const& kvDesc = inputDesc[3];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        const int32_t B = gr00t::v1::plugins::sdpa_sample_count(M, S);
        if (B < 0) return -1;  // rows do not tile S: malformed (..., S, K) input
        int32_t K = xDesc.dims.d[nbDims - 1];

        int64_t MEnc = 1;
        for (int32_t i = 0; i < kvDesc.dims.nbDims - 1; ++i) MEnc *= kvDesc.dims.d[i];
        const int32_t H = mNumHeads;
        const int32_t D = mHeadDim;

        // Workspace partition (no kv_bf16; that's in inputs[3])
        size_t x_i8_sz    = alignUp(static_cast<size_t>(M) * K);
        size_t x_sc_sz    = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t q_bf16_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        // MEnc already folds in the batch factor (product of the KV cache's leading
        // dims), so multiplying by B over-counts and pushes every buffer after
        // scores past the workspace getWorkspaceSize granted (numHeads*S*MEnc).
        size_t scores_sz  = alignUp(static_cast<size_t>(H) * S * MEnc * sizeof(uint16_t));
        size_t attn_b_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t attn_i8_sz = alignUp(static_cast<size_t>(M) * mInnerDim);

        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* x_i8     = reinterpret_cast<int8_t*>(ws);
        float*  x_sc     = reinterpret_cast<float*>(ws + x_i8_sz);
        void*   q_bf16   = static_cast<void*>(ws + x_i8_sz + x_sc_sz);
        void*   scores   = static_cast<void*>(static_cast<uint8_t*>(q_bf16) + q_bf16_sz);
        void*   attn_b   = static_cast<void*>(static_cast<uint8_t*>(scores) + scores_sz);
        int8_t* attn_i8  = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(attn_b) + attn_b_sz);
        float*  attn_sc  = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(attn_i8) + attn_i8_sz);

        // --- Step 1: AdaLN + per-row INT8 quant on x ---
        int rc;
        if (mStaticActScaleXDevice != nullptr &&
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

        // --- Step 3: SKIP. K, V come from inputs[3] (cached kv_bf16) ---

        // --- Steps 4-6: SDPA (Q·Kᵀ, masked softmax, P·V) ---
        // q_bf16 is (B, S, H*D); the cached inputs[3] is (B, S_enc, 2*H*D) with K at
        // offset 0 and V at offset H*D; attn_b is (B, S, H*D).
        if (B <= 0 || MEnc <= 0 || MEnc % B != 0) return -1;
        const int32_t SEnc = static_cast<int32_t>(MEnc / B);

        using gr00t::v1::plugins::SdpaOperand;
        auto* qBF16 = reinterpret_cast<__nv_bfloat16*>(q_bf16);
        auto const* kvBF16 = reinterpret_cast<__nv_bfloat16 const*>(inputs[3]);
        const int ld_q = mInnerDim;
        const int ld_kv = 2 * mInnerDim;
        const long long q_sample = static_cast<long long>(S) * ld_q;
        const long long kv_sample = static_cast<long long>(SEnc) * ld_kv;
        const long long attn_sample = static_cast<long long>(S) * mInnerDim;

        // The mask is (B, 1, 1, S_enc): one additive row per sample.
        void const* mask_ptr = mHasMask ? inputs[4] : nullptr;
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

        // --- Step 7: per-row quant on attn_buf ---
        if (mStaticActScalePostSdpaDevice != nullptr &&
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

        // --- Step 8: INT8 attn_O GEMM + bias + residual ---
        rc = dit_int8_rowwise_gemm_bias_residual_bf16out(
            attn_i8, mOI8Device,
            attn_sc, mOScaleDevice, mOBiasDevice,
            inputs[0],
            outputs[0],
            static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedCrossAttnFullCachedPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t nbInputs,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    mHasMask = (nbInputs == 5);
    return 0;
}

IPluginV3* FusedCrossAttnFullCachedPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedCrossAttnFullCachedPlugin::getFieldsToSerialize() noexcept {
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
    pushI8("weight_o_i8",     mOI8Host);
    pushF32("weight_o_scale",  mOScaleHost);
    pushF32("bias_o",          mOBiasHost);
    if (!mStaticActScaleXHost.empty())        pushF32("static_act_scale_x",         mStaticActScaleXHost);
    if (!mStaticActScalePostSdpaHost.empty()) pushF32("static_act_scale_post_sdpa", mStaticActScalePostSdpaHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("num_heads", &mNumHeads, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("head_dim",  &mHeadDim,  PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",       &mEps,      PluginFieldType::kFLOAT32, 1));
    mHasMaskSerialized = mHasMask ? 1 : 0;
    mDataToSerialize.emplace_back(PluginField("has_mask",  &mHasMaskSerialized, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedCrossAttnFullCachedPluginCreator::FusedCrossAttnFullCachedPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_q_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_q_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_q", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_o_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_post_sdpa", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedCrossAttnFullCachedPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedCrossAttnFullCachedPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedCrossAttnFullCachedPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedCrossAttnFullCachedPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedCrossAttnFullCachedPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedCrossAttnFullCachedPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedCrossAttnFullCachedPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedCrossAttnFullCached(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedCrossAttnFullCachedPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedCrossAttnFullCachedPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedCrossAttnFullCachedPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
