// Cross-attn prologue plugin with pre-quantized encoder.
//
// Structurally identical to FusedAdaLnQuantCrossAttn except step 2 (encoder
// per-row quant) is replaced by reading the (int8_enc, scale_enc) plugin
// inputs from a single upstream EncoderPreQuant node.

#include "plugin_field_util.h"
#include "fused_cross_attn_prequantized_plugin.h"
#include "fused_adaln_quant.h"
#include "dit_int8_rowwise_v2_fused.h"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedAdaLnQuantCrossAttnPrequantized"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace: x_i8 + x_scale + merged_KV. Encoder is provided externally.
inline size_t workspaceBytes(int32_t Mx, int32_t K, int32_t MEnc, int32_t innerDim) {
    size_t x_i8 = alignUp(static_cast<size_t>(Mx) * K);
    size_t x_sc = alignUp(static_cast<size_t>(Mx) * sizeof(float));
    size_t merged = alignUp(static_cast<size_t>(MEnc) * 2 * static_cast<size_t>(innerDim) * sizeof(uint16_t));
    return x_i8 + x_sc + merged;
}
}  // anon

PluginFieldCollection FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedAdaLnQuantCrossAttnPrequantizedPluginCreator);

FusedAdaLnQuantCrossAttnPrequantizedPlugin::FusedAdaLnQuantCrossAttnPrequantizedPlugin(
    std::string const& name,
    std::vector<int8_t> wQ, std::vector<float> wQScale, std::vector<float> bQ,
    std::vector<int8_t> wKV, std::vector<float> wKVScale, std::vector<float> bKV,
    int32_t innerDim, int32_t K, int32_t KEnc, float eps)
    : mLayerName(name)
    , mQI8Host(std::move(wQ))
    , mQScaleHost(std::move(wQScale))
    , mQBiasHost(std::move(bQ))
    , mKVI8Host(std::move(wKV))
    , mKVScaleHost(std::move(wKVScale))
    , mKVBiasHost(std::move(bKV))
    , mInnerDim(innerDim)
    , mK(K)
    , mKEnc(KEnc)
    , mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedAdaLnQuantCrossAttnPrequantizedPlugin::FusedAdaLnQuantCrossAttnPrequantizedPlugin(
    std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_q_i8")    { auto* p = static_cast<int8_t const*>(f.data); mQI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_q_scale") { auto* p = static_cast<float const*>(f.data); mQScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_q")    { auto* p = static_cast<float const*>(f.data); mQBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "weight_kv_i8") { auto* p = static_cast<int8_t const*>(f.data); mKVI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_kv_scale") { auto* p = static_cast<float const*>(f.data); mKVScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_kv")   { auto* p = static_cast<float const*>(f.data); mKVBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_x") { auto* p = static_cast<float const*>(f.data); mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "inner_dim") { mInnerDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "K")         { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "K_enc")     { mKEnc = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")       { mEps = *static_cast<float const*>(f.data); }
    }
}

FusedAdaLnQuantCrossAttnPrequantizedPlugin::~FusedAdaLnQuantCrossAttnPrequantizedPlugin() {
    if (mQI8Device)              cudaFree(mQI8Device);
    if (mQScaleDevice)           cudaFree(mQScaleDevice);
    if (mQBiasDevice)            cudaFree(mQBiasDevice);
    if (mKVI8Device)             cudaFree(mKVI8Device);
    if (mKVScaleDevice)          cudaFree(mKVScaleDevice);
    if (mKVBiasDevice)           cudaFree(mKVBiasDevice);
    if (mStaticActScaleXDevice)  cudaFree(mStaticActScaleXDevice);
}

void FusedAdaLnQuantCrossAttnPrequantizedPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mQI8Device,     mQI8Host.data(),     mQI8Host.size());
    upload(&mQScaleDevice,  mQScaleHost.data(),  mQScaleHost.size()  * sizeof(float));
    upload(&mQBiasDevice,   mQBiasHost.data(),   mQBiasHost.size()   * sizeof(float));
    upload(&mKVI8Device,    mKVI8Host.data(),    mKVI8Host.size());
    upload(&mKVScaleDevice, mKVScaleHost.data(), mKVScaleHost.size() * sizeof(float));
    upload(&mKVBiasDevice,  mKVBiasHost.data(),  mKVBiasHost.size()  * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
}

IPluginCapability* FusedAdaLnQuantCrossAttnPrequantizedPlugin::getCapabilityInterface(
    PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedAdaLnQuantCrossAttnPrequantizedPlugin::clone() noexcept {
    try {
        auto* p = new FusedAdaLnQuantCrossAttnPrequantizedPlugin(
            mLayerName,
            mQI8Host, mQScaleHost, mQBiasHost,
            mKVI8Host, mKVScaleHost, mKVBiasHost,
            mInnerDim, mK, mKEnc, mEps);
        p->mStaticActScaleXHost = mStaticActScaleXHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedAdaLnQuantCrossAttnPrequantizedPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedAdaLnQuantCrossAttnPrequantizedPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedAdaLnQuantCrossAttnPrequantizedPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedAdaLnQuantCrossAttnPrequantizedPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::getNbOutputs() const noexcept { return 3; }

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 3);
    outputTypes[0] = DataType::kBF16;
    outputTypes[1] = DataType::kBF16;
    outputTypes[2] = DataType::kBF16;
    return 0;
}

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::getOutputShapes(
    DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 5 && nbOutputs == 3);
    // Q: [B, S, inner_dim] from x (inputs[0]).
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[0].d[i] = inputs[0].d[i];
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mInnerDim);
    // K, V: [B, S_enc, inner_dim] from enc_i8 (inputs[3]).
    for (int32_t out = 1; out < 3; ++out) {
        outputs[out].nbDims = inputs[3].nbDims;
        for (int32_t i = 0; i < inputs[3].nbDims - 1; ++i) outputs[out].d[i] = inputs[3].d[i];
        outputs[out].d[inputs[3].nbDims - 1] = exprBuilder.constant(mInnerDim);
    }
    return 0;
}

bool FusedAdaLnQuantCrossAttnPrequantizedPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 5 && nbOutputs == 3);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    // pos 0,1,2: x, scale, shift → BF16
    if (pos == 0 || pos == 1 || pos == 2) return d.type == DataType::kBF16;
    // pos 3: enc_i8 → INT32-packed (4 INT8/element), to bypass TRT INT8 BuilderFlag requirement
    if (pos == 3) return d.type == DataType::kINT32;
    // pos 4: enc_scale → FP32
    if (pos == 4) return d.type == DataType::kFLOAT;
    // pos 5,6,7: outputs Q, K, V → BF16
    if (pos >= 5 && pos < 8) return d.type == DataType::kBF16;
    return false;
}

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mx = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mx *= inputs[0].max.d[i];
    int32_t K = inputs[0].max.d[inputs[0].max.nbDims - 1];
    int64_t MEnc = 1;
    for (int32_t i = 0; i < inputs[3].max.nbDims - 1; ++i) MEnc *= inputs[3].max.d[i];
    return workspaceBytes(static_cast<int32_t>(Mx), K, static_cast<int32_t>(MEnc), mInnerDim);
}

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();

        auto const& xDesc   = inputDesc[0];
        auto const& encDesc = inputDesc[3];

        int32_t S = xDesc.dims.d[xDesc.dims.nbDims - 2];
        int64_t Mx = 1;
        for (int32_t i = 0; i < xDesc.dims.nbDims - 1; ++i) Mx *= xDesc.dims.d[i];
        int32_t B = static_cast<int32_t>(Mx / S);
        int32_t K = xDesc.dims.d[xDesc.dims.nbDims - 1];

        int64_t MEnc = 1;
        for (int32_t i = 0; i < encDesc.dims.nbDims - 1; ++i) MEnc *= encDesc.dims.d[i];
        // Encoder INT8 is delivered packed as INT32 (4 INT8/elem), last dim = K_enc / 4.
        // Use the plugin field for the true KEnc.
        int32_t KEnc = mKEnc;

        size_t x_i8_sz = alignUp(static_cast<size_t>(Mx) * K);
        size_t x_sc_sz = alignUp(static_cast<size_t>(Mx) * sizeof(float));
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* x_i8  = reinterpret_cast<int8_t*>(ws);
        float*  x_sc  = reinterpret_cast<float*>(ws + x_i8_sz);
        void*   merged = static_cast<void*>(ws + x_i8_sz + x_sc_sz);

        // Step 1: AdaLN + per-row quant on x.
        int rc;
        if (mStaticActScaleXDevice != nullptr &&
            (int)mStaticActScaleXHost.size() >= static_cast<int>(Mx)) {
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

        // Step 2: REUSE pre-quantized encoder from inputs[3] (INT8) and inputs[4] (FP32 scale).
        // No quant kernel call here; encoder INT8 is reused across blocks.
        void const* enc_i8 = inputs[3];
        void const* enc_sc = inputs[4];

        // Step 3: Q GEMM.
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            x_i8, mQI8Device,
            x_sc, mQScaleDevice, mQBiasDevice,
            outputs[0],
            static_cast<int32_t>(Mx), mInnerDim, K, stream);
        if (rc != 0) return rc;

        // Step 4: KV merged GEMM on pre-quantized encoder.
        int32_t Nkv = 2 * mInnerDim;
        rc = dit_int8_rowwise_gemm_bias_bf16out(
            enc_i8, mKVI8Device,
            enc_sc, mKVScaleDevice, mKVBiasDevice,
            merged,
            static_cast<int32_t>(MEnc), Nkv, KEnc, stream);
        if (rc != 0) return rc;

        // Step 5: split merged → K, V.
        size_t srcStride = static_cast<size_t>(Nkv) * sizeof(uint16_t);
        size_t dstStride = static_cast<size_t>(mInnerDim) * sizeof(uint16_t);
        auto const* mergedB = static_cast<uint8_t const*>(merged);
        for (int32_t which = 0; which < 2; ++which) {
            cudaError_t e = cudaMemcpy2DAsync(
                outputs[1 + which], dstStride,
                mergedB + which * dstStride, srcStride,
                dstStride, static_cast<size_t>(MEnc),
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) return static_cast<int>(e);
        }
        return 0;
    } catch (...) { return -1; }
}

int32_t FusedAdaLnQuantCrossAttnPrequantizedPlugin::onShapeChange(
    PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedAdaLnQuantCrossAttnPrequantizedPlugin::attachToContext(
    IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedAdaLnQuantCrossAttnPrequantizedPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    auto pushI8 = [&](const char* name, std::vector<int8_t>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kINT8,
                                                  static_cast<int32_t>(v.size())));
    };
    auto pushF32 = [&](const char* name, std::vector<float>& v) {
        mDataToSerialize.emplace_back(PluginField(name, v.data(), PluginFieldType::kFLOAT32,
                                                  static_cast<int32_t>(v.size())));
    };
    pushI8("weight_q_i8",     mQI8Host);
    pushF32("weight_q_scale", mQScaleHost);
    pushF32("bias_q",         mQBiasHost);
    pushI8("weight_kv_i8",    mKVI8Host);
    pushF32("weight_kv_scale",mKVScaleHost);
    pushF32("bias_kv",        mKVBiasHost);
    if (!mStaticActScaleXHost.empty()) pushF32("static_act_scale_x", mStaticActScaleXHost);
    mDataToSerialize.emplace_back(PluginField("inner_dim", &mInnerDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K",         &mK,        PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K_enc",     &mKEnc,     PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",       &mEps,      PluginFieldType::kFLOAT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::FusedAdaLnQuantCrossAttnPrequantizedPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_q_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_q_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_q", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_kv_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_kv", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("inner_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K_enc", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::getPluginName() const noexcept { return kPLUGIN_NAME; }
char const* FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedAdaLnQuantCrossAttnPrequantizedPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedAdaLnQuantCrossAttnPrequantizedPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedCrossAttnPrequantized(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedAdaLnQuantCrossAttnPrequantizedPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedCrossAttnPrequantizedPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedAdaLnQuantCrossAttnPrequantizedPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
