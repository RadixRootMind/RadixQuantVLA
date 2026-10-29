// FusedNormProjOut: norm_out + AdaLN + per-row INT8 quant + INT8 proj_out_2 GEMM + bias.
//
// Reuses existing kernels:
//   fused_adaln_static_quant_bf16_to_int8 / fused_adaln_quant_bf16_to_int8, for LN+AdaLN+quant
//   dit_int8_rowwise_gemm_bias_bf16out, for the proj_out_2 INT8 GEMM with bias

#include "plugin_field_util.h"
#include "fused_norm_projout_plugin.h"
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
constexpr char const* kPLUGIN_NAME{"FusedNormProjOut"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t alignUp(size_t v) { return (v + 127) & ~static_cast<size_t>(127); }

// Workspace: x_i8 (M*K) + x_scale (M*4)
inline size_t workspaceBytes(int32_t M, int32_t K) {
    size_t a = alignUp(static_cast<size_t>(M) * K);
    size_t s = alignUp(static_cast<size_t>(M) * sizeof(float));
    return a + s;
}
}  // anon

PluginFieldCollection FusedNormProjOutPluginCreator::mFieldCollection{};
std::vector<PluginField> FusedNormProjOutPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedNormProjOutPluginCreator);

FusedNormProjOutPlugin::FusedNormProjOutPlugin(std::string const& name,
    std::vector<int8_t> wI8, std::vector<float> wScale, std::vector<float> bias,
    int32_t K, int32_t outputDim, float eps)
    : mLayerName(name)
    , mWI8Host(std::move(wI8))
    , mWScaleHost(std::move(wScale))
    , mBiasHost(std::move(bias))
    , mK(K), mOutputDim(outputDim), mEps(eps) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedNormProjOutPlugin::FusedNormProjOutPlugin(std::string const& name,
    PluginFieldCollection const* fc) : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if      (n == "weight_proj_out_i8")    { auto* p = static_cast<int8_t const*>(f.data); mWI8Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_proj_out_scale") { auto* p = static_cast<float const*>(f.data);  mWScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "bias_proj_out")         { auto* p = static_cast<float const*>(f.data);  mBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "static_act_scale_x")    { auto* p = static_cast<float const*>(f.data);  mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "K")          { mK = *static_cast<int32_t const*>(f.data); }
        else if (n == "output_dim") { mOutputDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "eps")        { mEps = *static_cast<float const*>(f.data); }
    }
}

FusedNormProjOutPlugin::~FusedNormProjOutPlugin() {
    if (mWI8Device)            cudaFree(mWI8Device);
    if (mWScaleDevice)         cudaFree(mWScaleDevice);
    if (mBiasDevice)           cudaFree(mBiasDevice);
    if (mStaticActScaleXDevice) cudaFree(mStaticActScaleXDevice);
}

void FusedNormProjOutPlugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    upload(&mWI8Device,    mWI8Host.data(),    mWI8Host.size());
    upload(&mWScaleDevice, mWScaleHost.data(), mWScaleHost.size() * sizeof(float));
    upload(&mBiasDevice,   mBiasHost.data(),   mBiasHost.size()   * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
}

IPluginCapability* FusedNormProjOutPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedNormProjOutPlugin::clone() noexcept {
    try {
        auto* p = new FusedNormProjOutPlugin(
            mLayerName, mWI8Host, mWScaleHost, mBiasHost, mK, mOutputDim, mEps);
        p->mStaticActScaleXHost = mStaticActScaleXHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedNormProjOutPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedNormProjOutPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedNormProjOutPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedNormProjOutPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedNormProjOutPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedNormProjOutPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedNormProjOutPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    // out: [B, S, output_dim]; preserve leading dims, replace last with output_dim.
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[0].d[i] = inputs[0].d[i];
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mOutputDim);
    return 0;
}

bool FusedNormProjOutPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 3 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedNormProjOutPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedNormProjOutPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax);
}

int32_t FusedNormProjOutPlugin::enqueue(PluginTensorDesc const* inputDesc,
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

        size_t x_i8_sz = alignUp(static_cast<size_t>(M) * K);
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* x_i8 = reinterpret_cast<int8_t*>(ws);
        float*  x_sc = reinterpret_cast<float*>(ws + x_i8_sz);

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

        // INT8 GEMM proj_out_2 with bias → BF16 output (no residual).
        return dit_int8_rowwise_gemm_bias_bf16out(
            x_i8, mWI8Device,
            x_sc, mWScaleDevice, mBiasDevice,
            outputs[0],
            static_cast<int32_t>(M), mOutputDim, K, stream);
    } catch (...) { return -1; }
}

int32_t FusedNormProjOutPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedNormProjOutPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedNormProjOutPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("weight_proj_out_i8",
        mWI8Host.data(), PluginFieldType::kINT8, static_cast<int32_t>(mWI8Host.size())));
    mDataToSerialize.emplace_back(PluginField("weight_proj_out_scale",
        mWScaleHost.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mWScaleHost.size())));
    mDataToSerialize.emplace_back(PluginField("bias_proj_out",
        mBiasHost.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mBiasHost.size())));
    if (!mStaticActScaleXHost.empty()) {
        mDataToSerialize.emplace_back(PluginField("static_act_scale_x",
            mStaticActScaleXHost.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mStaticActScaleXHost.size())));
    }
    mDataToSerialize.emplace_back(PluginField("K",          &mK,         PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("output_dim", &mOutputDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps",        &mEps,       PluginFieldType::kFLOAT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FusedNormProjOutPluginCreator::FusedNormProjOutPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_proj_out_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_proj_out_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("bias_proj_out", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedNormProjOutPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedNormProjOutPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedNormProjOutPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedNormProjOutPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedNormProjOutPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedNormProjOutPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedNormProjOutPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedNormProjOut(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedNormProjOutPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedNormProjOutPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedNormProjOutPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
