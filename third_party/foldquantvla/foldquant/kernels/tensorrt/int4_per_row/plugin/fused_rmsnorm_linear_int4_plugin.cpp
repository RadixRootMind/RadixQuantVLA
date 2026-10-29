// IPluginV3 wrapper: RMSNorm + FWHT + per-row INT4 quant + INT4 Linear (no bias).
// LLM W4A4 merged-GEMM block: used for Q+K+V and gate+up.

#include "plugin_field_util.h"
#include "fused_rmsnorm_linear_int4_plugin.h"
#include "rmsnorm_per_row_quant_int4.h"  // RMSNorm + FWHT + per-row int4 prologue
#include "dit_int4_rowwise.h"            // s4 GEMM

#include <cassert>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedRmsNormLinearInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

// Workspace: [packed INT4 act (M*K/2 bytes)] + [per-row scale (M*4 bytes)].
// enqueue() slices the same buffer, so both must agree on where the scale
// starts, hence one definition of the offset rather than two.
inline size_t actBytes(int32_t M, int32_t K) {
    return (static_cast<size_t>(M) * (K / 2) + 127) & ~static_cast<size_t>(127);
}
inline size_t workspaceBytes(int32_t M, int32_t K) {
    size_t a = actBytes(M, K);
    size_t s = (static_cast<size_t>(M) * sizeof(float) + 127) & ~static_cast<size_t>(127);
    return a + s;
}
}  // anon

PluginFieldCollection FusedRmsNormLinearInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> FusedRmsNormLinearInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedRmsNormLinearInt4PluginCreator);

FusedRmsNormLinearInt4Plugin::FusedRmsNormLinearInt4Plugin(
    std::string const& name,
    std::vector<int8_t> weightI4,
    std::vector<float>  weightScale,
    std::vector<uint16_t> gammaBf16,
    int32_t N, int32_t K, float eps, int32_t rotBlockSize, float actClipRatio)
    : mLayerName(name)
    , mWeightI4Host(std::move(weightI4))
    , mWeightScaleHost(std::move(weightScale))
    , mGammaBf16Host(std::move(gammaBf16))
    , mN(N), mK(K), mEps(eps), mRotBlockSize(rotBlockSize), mActClipRatio(actClipRatio) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedRmsNormLinearInt4Plugin::FusedRmsNormLinearInt4Plugin(
    std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        if (n == "weight_i4") {
            auto const* p = static_cast<int8_t const*>(fc->fields[i].data);
            mWeightI4Host.assign(p, p + gr00t::fieldElemCount(fc->fields[i], 1));
        } else if (n == "weight_scale") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mWeightScaleHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        } else if (n == "gamma") {
            // BF16 transported as a uint16_t array; fieldElemCount normalizes
            // the parser's byte length vs the deserializer's element count.
            auto const* p = static_cast<uint16_t const*>(fc->fields[i].data);
            mGammaBf16Host.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(uint16_t)));
        } else if (n == "N") {
            mN = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "K") {
            mK = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "eps") {
            mEps = *static_cast<float const*>(fc->fields[i].data);
        } else if (n == "act_clip_ratio") {
            mActClipRatio = *static_cast<float const*>(fc->fields[i].data);
        } else if (n == "rot_block_size") {
            mRotBlockSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

FusedRmsNormLinearInt4Plugin::~FusedRmsNormLinearInt4Plugin() {
    if (mWeightI4Device)         cudaFree(mWeightI4Device);
    if (mWeightScaleDevice)      cudaFree(mWeightScaleDevice);
    if (mGammaDevice)            cudaFree(mGammaDevice);
    if (mZeroBiasDevice)         cudaFree(mZeroBiasDevice);
}

int FusedRmsNormLinearInt4Plugin::ensureWeightsOnDevice() {
    // Propagate allocation/copy failures: on OOM a swallowed error would run
    // the GEMM with a null weight pointer and surface as a sticky device
    // fault misattributed to whatever enqueues next.
    auto upload = [](void** dev, const void* host, size_t bytes) -> cudaError_t {
        if (*dev || bytes == 0) return cudaSuccess;
        cudaError_t rc = cudaMalloc(dev, bytes);
        if (rc != cudaSuccess) return rc;
        return cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    cudaError_t rc;
    if ((rc = upload(&mWeightI4Device,    mWeightI4Host.data(),    mWeightI4Host.size())) != cudaSuccess) return rc;
    if ((rc = upload(&mWeightScaleDevice, mWeightScaleHost.data(), mWeightScaleHost.size() * sizeof(float))) != cudaSuccess) return rc;
    if ((rc = upload(&mGammaDevice,       mGammaBf16Host.data(),   mGammaBf16Host.size()  * sizeof(uint16_t))) != cudaSuccess) return rc;
    if (!mZeroBiasDevice && mN > 0) {
        if ((rc = cudaMalloc(&mZeroBiasDevice, static_cast<size_t>(mN) * sizeof(float))) != cudaSuccess) return rc;
        if ((rc = cudaMemset(mZeroBiasDevice, 0, static_cast<size_t>(mN) * sizeof(float))) != cudaSuccess) return rc;
    }
    return cudaSuccess;
}

IPluginCapability* FusedRmsNormLinearInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedRmsNormLinearInt4Plugin::clone() noexcept {
    try {
        auto* p = new FusedRmsNormLinearInt4Plugin(
            mLayerName, mWeightI4Host, mWeightScaleHost, mGammaBf16Host, mN, mK,
            mEps, mRotBlockSize, mActClipRatio);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedRmsNormLinearInt4Plugin::getPluginName() const noexcept     { return kPLUGIN_NAME; }
char const* FusedRmsNormLinearInt4Plugin::getPluginVersion() const noexcept  { return kPLUGIN_VERSION; }
char const* FusedRmsNormLinearInt4Plugin::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void FusedRmsNormLinearInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedRmsNormLinearInt4Plugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedRmsNormLinearInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedRmsNormLinearInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) {
        outputs[0].d[i] = inputs[0].d[i];
    }
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mN);
    return 0;
}

bool FusedRmsNormLinearInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedRmsNormLinearInt4Plugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedRmsNormLinearInt4Plugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax);
}

int32_t FusedRmsNormLinearInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        int wrc = ensureWeightsOnDevice();
        if (wrc != 0) return wrc;

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int32_t S = (nbDims >= 2) ? xDesc.dims.d[nbDims - 2] : 1;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        int32_t B = static_cast<int32_t>(M / S);
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);

        size_t aBytes = actBytes(static_cast<int32_t>(M), K);
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* actI4    = reinterpret_cast<int8_t*>(ws);   // packed, 2 nibbles/byte
        float*  actScale = reinterpret_cast<float*>(ws + aBytes);

        // Step 1: RMSNorm + optional block-Hadamard + per-row INT4 quant.
        // Dynamic only; see the header for why static scales are absent here.
        int rc = rmsnorm_fwht_per_row_quant_bf16_to_int4(
            inputs[0], mGammaDevice,
            actI4, actScale,
            B, S, K, mEps, mRotBlockSize, mActClipRatio, stream);
        if (rc != 0) return rc;

        // Step 2: INT4 GEMM, native s4 tensor-core path
        // (mma.sync.m16n8k64.s4.s4.s32), per-row act scale x per-col weight
        // scale. The bias operand is a zero vector: Qwen3 has no bias here and
        // the no-bias entry point in dit_int4_rowwise.h has no implementation
        // (see mZeroBiasDevice in the header).
        rc = dit_int4_rowwise_gemm_bias_bf16out(
            actI4, mWeightI4Device,
            actScale, mWeightScaleDevice,
            mZeroBiasDevice,
            outputs[0],
            static_cast<int32_t>(M), mN, K, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedRmsNormLinearInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedRmsNormLinearInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedRmsNormLinearInt4Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        PluginField("weight_i4", mWeightI4Host.data(), PluginFieldType::kINT8,
                    static_cast<int32_t>(mWeightI4Host.size())));
    mDataToSerialize.emplace_back(
        PluginField("weight_scale", mWeightScaleHost.data(), PluginFieldType::kFLOAT32,
                    static_cast<int32_t>(mWeightScaleHost.size())));
    mDataToSerialize.emplace_back(
        PluginField("gamma", mGammaBf16Host.data(), PluginFieldType::kBF16,
                    static_cast<int32_t>(mGammaBf16Host.size())));
    mDataToSerialize.emplace_back(PluginField("N", &mN, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1));
    // MUST be serialized: configurePlugin does not re-run on deserialize, and
    // the baked weights are already folded with W·Hᵀ.
    mDataToSerialize.emplace_back(
        PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    // A lost clip ratio silently changes the activation grid the GPTQ weights
    // were rounded against, so serialize it like rot_block_size.
    mDataToSerialize.emplace_back(
        PluginField("act_clip_ratio", &mActClipRatio, PluginFieldType::kFLOAT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// Creator
// ============================================================================

FusedRmsNormLinearInt4PluginCreator::FusedRmsNormLinearInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_clip_ratio", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedRmsNormLinearInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedRmsNormLinearInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedRmsNormLinearInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedRmsNormLinearInt4PluginCreator::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void FusedRmsNormLinearInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedRmsNormLinearInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedRmsNormLinearInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedRmsNormLinearInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedRmsNormLinearInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedRmsNormLinearInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedRmsNormLinearInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
