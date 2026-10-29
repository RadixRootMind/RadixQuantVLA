// FusedSelfAttnFull: AdaLN + INT8 merged QKV + cuBLAS BF16 SDPA + INT8 attn_O
// + bias + residual collapsed into one plugin call.

#include "plugin_field_util.h"
#include "fused_selfattn_full_plugin.h"
#include "fused_adaln_quant.h"
#include "dit_int8_rowwise_v2.h"          // per_row_quant
#include "dit_int8_rowwise_v2_fused.h"    // gemm_bias, gemm_bias_residual
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
constexpr char const* kPLUGIN_NAME{"FusedSelfAttnFull"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace:
//   [x_i8 (M*K)]               INT8 - post-AdaLN quantized input
//   [x_scale (M*4)]             FP32 - per-row scale of x
//   [merged_qkv (M*3*inner*2)] BF16: output of merged QKV GEMM, laid out (S, 3, H, D)
//   [scores (H*S*S*2)]         BF16 - Q·Kᵀ result, then softmax in-place
//   [attn_buf (M*inner*2)]     BF16 - scores·V result, laid out (S, H, D)
//   [attn_i8 (M*inner)]        INT8 - post-SDPA per-row quantized
//   [attn_scale (M*4)]         FP32 - per-row scale
inline size_t workspaceBytes(int32_t M, int32_t K, int32_t innerDim,
                              int32_t numHeads, int32_t S) {
    size_t x_i8    = alignUp(static_cast<size_t>(M) * K);
    size_t x_sc    = alignUp(static_cast<size_t>(M) * sizeof(float));
    size_t merged  = alignUp(static_cast<size_t>(M) * 3 * static_cast<size_t>(innerDim) * sizeof(uint16_t));
    // scores is laid out B*H*S*S in enqueue(); M == B*S, so B == M/S. Omitting the
    // batch factor here under-allocates everything placed after scores.
    size_t B       = (S > 0) ? static_cast<size_t>(M) / static_cast<size_t>(S) : 1;
    size_t scores  = alignUp(B * static_cast<size_t>(numHeads) * static_cast<size_t>(S) * S * sizeof(uint16_t));
    size_t attn_b  = alignUp(static_cast<size_t>(M) * innerDim * sizeof(uint16_t));
    size_t attn_i8 = alignUp(static_cast<size_t>(M) * innerDim);
    size_t attn_sc = alignUp(static_cast<size_t>(M) * sizeof(float));
    return x_i8 + x_sc + merged + scores + attn_b + attn_i8 + attn_sc;
}
}  // anon

PluginFieldCollection FusedSelfAttnFullPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedSelfAttnFullPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedSelfAttnFullPluginCreator);

FusedSelfAttnFullPlugin::FusedSelfAttnFullPlugin(std::string const& name,
    std::vector<int8_t> wQKV, std::vector<float> wQKVScale, std::vector<float> bQKV,
    std::vector<int8_t> wO,   std::vector<float> wOScale,   std::vector<float> bO,
    int32_t innerDim, int32_t K, int32_t numHeads, int32_t headDim, float eps)
    : mLayerName(name)
    , mQKVI8Host(std::move(wQKV))
    , mQKVScaleHost(std::move(wQKVScale))
    , mQKVBiasHost(std::move(bQKV))
    , mOI8Host(std::move(wO))
    , mOScaleHost(std::move(wOScale))
    , mOBiasHost(std::move(bO))
    , mInnerDim(innerDim)
    , mK(K)
    , mNumHeads(numHeads)
    , mHeadDim(headDim)
    , mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedSelfAttnFullPlugin::FusedSelfAttnFullPlugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_qkv_i8")    { auto* p = static_cast<int8_t const*>(f.data); mQKVI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_qkv_scale") { auto* p = static_cast<float const*>(f.data); mQKVScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_qkv")    { auto* p = static_cast<float const*>(f.data); mQKVBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_o_i8") { auto* p = static_cast<int8_t const*>(f.data); mOI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_o_scale") { auto* p = static_cast<float const*>(f.data); mOScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_o")      { auto* p = static_cast<float const*>(f.data); mOBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_x") { auto* p = static_cast<float const*>(f.data); mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_post_sdpa") { auto* p = static_cast<float const*>(f.data); mStaticActScalePostSdpaHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "inner_dim") { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")         { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "num_heads") { mNumHeads = *static_cast<int32_t const*>(f.data); }
        else if (n == "head_dim")  { mHeadDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")       { mEps = *static_cast<float const*>(f.data); }
        else if (n == "rot_block_size")  { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre_in") { auto* p = static_cast<float const*>(f.data); mScalePreInHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "act_scale_pre_o")  { auto* p = static_cast<float const*>(f.data); mScalePreOHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
    }
}

FusedSelfAttnFullPlugin::~FusedSelfAttnFullPlugin() {
    if (mQKVI8Device)            cudaFree(mQKVI8Device);
    if (mQKVScaleDevice)         cudaFree(mQKVScaleDevice);
    if (mQKVBiasDevice)          cudaFree(mQKVBiasDevice);
    if (mOI8Device)              cudaFree(mOI8Device);
    if (mOScaleDevice)           cudaFree(mOScaleDevice);
    if (mOBiasDevice)            cudaFree(mOBiasDevice);
    if (mStaticActScaleXDevice)  cudaFree(mStaticActScaleXDevice);
    if (mStaticActScalePostSdpaDevice) cudaFree(mStaticActScalePostSdpaDevice);
    if (mScalePreInDevice) cudaFree(mScalePreInDevice);
    if (mScalePreODevice)  cudaFree(mScalePreODevice);
    if (mCublasHandle) cublasDestroy(reinterpret_cast<cublasHandle_t>(mCublasHandle));
}

void FusedSelfAttnFullPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mScalePreInDevice, mScalePreInHost.data(), mScalePreInHost.size() * sizeof(float));
    upload(&mScalePreODevice,  mScalePreOHost.data(),  mScalePreOHost.size()  * sizeof(float));
    upload(&mQKVI8Device,    mQKVI8Host.data(),    mQKVI8Host.size());
    upload(&mQKVScaleDevice, mQKVScaleHost.data(), mQKVScaleHost.size() * sizeof(float));
    upload(&mQKVBiasDevice,  mQKVBiasHost.data(),  mQKVBiasHost.size()  * sizeof(float));
    upload(&mOI8Device,      mOI8Host.data(),      mOI8Host.size());
    upload(&mOScaleDevice,   mOScaleHost.data(),   mOScaleHost.size()   * sizeof(float));
    upload(&mOBiasDevice,    mOBiasHost.data(),    mOBiasHost.size()    * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
    upload(&mStaticActScalePostSdpaDevice, mStaticActScalePostSdpaHost.data(),
           mStaticActScalePostSdpaHost.size() * sizeof(float));
}

void FusedSelfAttnFullPlugin::ensureCublasHandle() {
    if (mCublasHandle) return;
    cublasHandle_t h;
    cublasCreate(&h);
    mCublasHandle = reinterpret_cast<void*>(h);
}

IPluginCapability* FusedSelfAttnFullPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedSelfAttnFullPlugin::clone() noexcept {
    try {
        auto* p = new FusedSelfAttnFullPlugin(
            mLayerName,
            mQKVI8Host, mQKVScaleHost, mQKVBiasHost,
            mOI8Host,   mOScaleHost,   mOBiasHost,
            mInnerDim, mK, mNumHeads, mHeadDim, mEps);
        p->mStaticActScaleXHost         = mStaticActScaleXHost;
        p->mStaticActScalePostSdpaHost  = mStaticActScalePostSdpaHost;
        // The clone is what runs; a member dropped here silently stops the fold.
        p->mRotBlockSize    = mRotBlockSize;
        p->mScalePreInHost  = mScalePreInHost;
        p->mScalePreOHost   = mScalePreOHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedSelfAttnFullPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedSelfAttnFullPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedSelfAttnFullPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedSelfAttnFullPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedSelfAttnFullPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedSelfAttnFullPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedSelfAttnFullPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    // Output same shape as input x [B, S, K].
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
    return 0;
}

bool FusedSelfAttnFullPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedSelfAttnFullPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedSelfAttnFullPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t S = inputs[0].max.d[inputs[0].max.nbDims - 2];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax, mInnerDim, mNumHeads, S);
}

int32_t FusedSelfAttnFullPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();
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
        assert(H * D == mInnerDim);

        // --- Workspace partition ---
        size_t x_i8_sz    = alignUp(static_cast<size_t>(M) * K);
        size_t x_sc_sz    = alignUp(static_cast<size_t>(M) * sizeof(float));
        size_t merged_sz  = alignUp(static_cast<size_t>(M) * 3 * static_cast<size_t>(mInnerDim) * sizeof(uint16_t));
        size_t scores_sz  = alignUp(static_cast<size_t>(B) * H * S * S * sizeof(uint16_t));
        size_t attn_b_sz  = alignUp(static_cast<size_t>(M) * mInnerDim * sizeof(uint16_t));
        size_t attn_i8_sz = alignUp(static_cast<size_t>(M) * mInnerDim);

        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* x_i8    = reinterpret_cast<int8_t*>(ws);
        float*  x_sc    = reinterpret_cast<float*>(ws + x_i8_sz);
        void*   merged  = static_cast<void*>(ws + x_i8_sz + x_sc_sz);
        void*   scores  = static_cast<void*>(static_cast<uint8_t*>(merged) + merged_sz);
        void*   attn_b  = static_cast<void*>(static_cast<uint8_t*>(scores) + scores_sz);
        int8_t* attn_i8 = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(attn_b) + attn_b_sz);
        float*  attn_sc = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(attn_i8) + attn_i8_sz);

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

        // --- Step 2: INT8 GEMM merged QKV. Output layout: (M, 3*H*D) row-major.
        //     Equivalently (S, 3, H, D) under the standard order Q,K,V → 3 × (S, H, D) tiles.
        int32_t Ntot = 3 * mInnerDim;
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            x_i8, mQKVI8Device,
            x_sc, mQKVScaleDevice, mQKVBiasDevice,
            merged,
            static_cast<int32_t>(M), Ntot, K, stream);
        if (rc != 0) return rc;

        // --- Steps 3-5: SDPA (Q·Kᵀ, softmax, P·V) ---
        // `merged` is (B, S, 3, H, D) row-major: head h of Q/K/V for sample b sits at
        // merged + b*S*3*H*D + t*H*D + h*D with leading dim 3*H*D. `attn_b` is
        // (B, S, H*D) with head h at attn_b + b*S*H*D + h*D.
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

        // --- Step 6: per-row INT8 quant on attn_buf (M, inner_dim) ---
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

        // --- Step 7: INT8 attn_O GEMM with bias + residual=x → outputs[0] ---
        rc = dit_int8_rowwise_gemm_bias_residual_bf16out(
            attn_i8, mOI8Device,
            attn_sc, mOScaleDevice, mOBiasDevice,
            inputs[0],            // residual = x input
            outputs[0],
            static_cast<int32_t>(M), K, mInnerDim, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedSelfAttnFullPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedSelfAttnFullPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedSelfAttnFullPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* n, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kINT8,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* n, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(n, v.data(), PluginFieldType::kFLOAT32,
                                                  static_cast<int32_t>(v.size())));
    };
    pushI8("weight_qkv_i8",    mQKVI8Host);
    pushF32("weight_qkv_scale", mQKVScaleHost);
    pushF32("bias_qkv",         mQKVBiasHost);
    pushI8("weight_o_i8",      mOI8Host);
    pushF32("weight_o_scale",   mOScaleHost);
    pushF32("bias_o",           mOBiasHost);
    if (!mStaticActScaleXHost.empty())        pushF32("static_act_scale_x",         mStaticActScaleXHost);
    if (!mStaticActScalePostSdpaHost.empty()) pushF32("static_act_scale_post_sdpa", mStaticActScalePostSdpaHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
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
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedSelfAttnFullPluginCreator::FusedSelfAttnFullPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_qkv_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_qkv_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_qkv", nullptr, PluginFieldType::kFLOAT32, 0));
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
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_in", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_o", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedSelfAttnFullPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedSelfAttnFullPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedSelfAttnFullPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedSelfAttnFullPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedSelfAttnFullPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedSelfAttnFullPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedSelfAttnFullPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedSelfAttnFull(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedSelfAttnFullPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedSelfAttnFullPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedSelfAttnFullPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
