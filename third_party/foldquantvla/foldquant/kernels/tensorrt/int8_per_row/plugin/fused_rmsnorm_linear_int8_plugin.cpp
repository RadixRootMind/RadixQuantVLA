// IPluginV3 wrapper: RMSNorm + per-row INT8 quant + INT8 Linear (no bias).
// LLM ladder L3+ basic block, used standalone for q/k/v/o_proj/gate/up_proj.

#include "plugin_field_util.h"
#include "int4_unpack_util.h"
#include "fused_rmsnorm_linear_int8_plugin.h"
#include "rmsnorm_per_row_quant.h"
#include "dit_int8_rowwise_v2.h"        // (default tile, fallback)
#include "dit_int8_rowwise_v2_tiles.h"  // t128x128x64 autotuned for LLM

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"FusedRmsNormLinearInt8"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

// Workspace: [INT8 quantized act (M*K)] + [per-row scale (M*4 bytes)].
inline size_t workspaceBytes(int32_t M, int32_t K) {
    size_t a = (static_cast<size_t>(M) * K + 127) & ~static_cast<size_t>(127);
    size_t s = (static_cast<size_t>(M) * sizeof(float) + 127) & ~static_cast<size_t>(127);
    return a + s;
}
}  // anon

PluginFieldCollection FusedRmsNormLinearInt8PluginCreator::mFieldCollection{};
std::vector<PluginField> FusedRmsNormLinearInt8PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedRmsNormLinearInt8PluginCreator);

FusedRmsNormLinearInt8Plugin::FusedRmsNormLinearInt8Plugin(
    std::string const& name,
    std::vector<int8_t> weightI8,
    std::vector<float>  weightScale,
    std::vector<uint16_t> gammaBf16,
    int32_t N, int32_t K, float eps, int32_t rotBlockSize)
    : mLayerName(name)
    , mWeightI8Host(std::move(weightI8))
    , mWeightScaleHost(std::move(weightScale))
    , mGammaBf16Host(std::move(gammaBf16))
    , mN(N), mK(K), mEps(eps), mRotBlockSize(rotBlockSize) {
    mNamespace = kPLUGIN_NAMESPACE;
}

FusedRmsNormLinearInt8Plugin::FusedRmsNormLinearInt8Plugin(
    std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        if (n == "weight_i8") {
            auto const* p = static_cast<int8_t const*>(fc->fields[i].data);
            mWeightI8Host.assign(p, p + gr00t::fieldElemCount(fc->fields[i], 1));
        } else if (n == "weight_i4") {
            // W4A8: packed nibbles, N*K/2 bytes. Kept packed on the host so the
            // serialized engine stays INT4-sized; unpacked once at upload.
            auto const* p = static_cast<uint8_t const*>(fc->fields[i].data);
            mWeightI4Host.assign(p, p + gr00t::fieldElemCount(fc->fields[i], 1));
        } else if (n == "weight_scale") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mWeightScaleHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        } else if (n == "gamma") {
            // BF16 transported as uint16_t array (length is in elements).
            auto const* p = static_cast<uint16_t const*>(fc->fields[i].data);
            mGammaBf16Host.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(uint16_t)));
        } else if (n == "static_act_scale_x") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        } else if (n == "N") {
            mN = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "K") {
            mK = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "eps") {
            mEps = *static_cast<float const*>(fc->fields[i].data);
        } else if (n == "rot_block_size") {
            mRotBlockSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

FusedRmsNormLinearInt8Plugin::~FusedRmsNormLinearInt8Plugin() {
    if (mWeightI8Device)         cudaFree(mWeightI8Device);
    if (mWeightScaleDevice)      cudaFree(mWeightScaleDevice);
    if (mGammaDevice)            cudaFree(mGammaDevice);
    if (mStaticActScaleXDevice)  cudaFree(mStaticActScaleXDevice);
}

void FusedRmsNormLinearInt8Plugin::ensureWeightsOnDevice() {
    auto upload = [](void** dev, const void* host, size_t bytes) {
        if (*dev || bytes == 0) return;
        cudaMalloc(dev, bytes);
        cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice);
    };
    if (!mWeightI8Device && mWeightI8Host.empty() && !mWeightI4Host.empty()) {
        std::vector<int8_t> const unpacked = gr00t::unpackInt4Nibbles(mWeightI4Host);
        upload(&mWeightI8Device, unpacked.data(), unpacked.size());
    }
    upload(&mWeightI8Device,    mWeightI8Host.data(),    mWeightI8Host.size());
    upload(&mWeightScaleDevice, mWeightScaleHost.data(), mWeightScaleHost.size() * sizeof(float));
    upload(&mGammaDevice,       mGammaBf16Host.data(),   mGammaBf16Host.size()  * sizeof(uint16_t));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
}

IPluginCapability* FusedRmsNormLinearInt8Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* FusedRmsNormLinearInt8Plugin::clone() noexcept {
    try {
        auto* p = new FusedRmsNormLinearInt8Plugin(
            mLayerName, mWeightI8Host, mWeightScaleHost, mGammaBf16Host, mN, mK,
            mEps, mRotBlockSize);
        p->mStaticActScaleXHost = mStaticActScaleXHost;
        p->mWeightI4Host = mWeightI4Host;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* FusedRmsNormLinearInt8Plugin::getPluginName() const noexcept     { return kPLUGIN_NAME; }
char const* FusedRmsNormLinearInt8Plugin::getPluginVersion() const noexcept  { return kPLUGIN_VERSION; }
char const* FusedRmsNormLinearInt8Plugin::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void FusedRmsNormLinearInt8Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t FusedRmsNormLinearInt8Plugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedRmsNormLinearInt8Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t FusedRmsNormLinearInt8Plugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
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

bool FusedRmsNormLinearInt8Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t FusedRmsNormLinearInt8Plugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t FusedRmsNormLinearInt8Plugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax);
}

int32_t FusedRmsNormLinearInt8Plugin::enqueue(PluginTensorDesc const* inputDesc,
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

        size_t aBytes = (static_cast<size_t>(M) * K + 127) & ~static_cast<size_t>(127);
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* actI8    = reinterpret_cast<int8_t*>(ws);
        float*  actScale = reinterpret_cast<float*>(ws + aBytes);

        // Step 1: RMSNorm + per-row INT8 quant.
        int rc;
        if (mStaticActScaleXDevice != nullptr &&
            static_cast<int>(mStaticActScaleXHost.size()) >= static_cast<int>(M)) {
            // Static scales are calibrated pre-rotation, so combining them with a
            // rotation would quantize rotated values against unrotated scales.
            // Fail loudly rather than emit a wrong engine.
            if (mRotBlockSize > 1) return -1;
            rc = rmsnorm_per_row_static_quant_bf16_to_int8(
                inputs[0], mGammaDevice, mStaticActScaleXDevice,
                actI8, actScale,
                B, S, K, mEps, stream);
        } else {
            rc = rmsnorm_fwht_per_row_quant_bf16_to_int8(
                inputs[0], mGammaDevice,
                actI8, actScale,
                B, S, K, mEps, mRotBlockSize, stream);
        }
        if (rc != 0) return rc;

        // Step 2: INT8 GEMM (no bias) with autotune-selected tile.
        // t128x128x64_w64x64x64_s4 wins 8/8 LLM shapes (Q+K+V, o, gate+up, down)
        // per scripts/.../test_llm_tile_autotune.py: 1.2-1.6× speedup vs
        // the previously used dit_int8_rowwise_gemm_bf16out entry.
        // That autotune ran on LLM prefill shapes, where M is in the hundreds.
        // An action expert drives the same plugin with M = action horizon, and
        // there the 128-row tile computes mostly padding and halves the CTA
        // count: measured 2.0-2.9x slower at M<=64 on SM89. The LLM keeps the
        // autotuned tile; forcing the small one on LLM shapes cost 1.5 ms of
        // backbone latency when this rule also keyed on N.
        // NB: dit_int8_rowwise_gemm_bf16out is NOT the small tile; it is another
        // 128x128x64 (dit_int8_rowwise_v2_cuda.cu). The 64-row tile is the
        // explicitly named variant from the tiles TU.
        bool const smallTile = (M <= 64);  // action-expert row counts only
        rc = smallTile
                 ? dit_int8_rowwise_gemm_bf16out_t64x64x64_w32x32x64_s4(
                       actI8, mWeightI8Device, actScale, mWeightScaleDevice,
                       outputs[0], static_cast<int32_t>(M), mN, K, stream)
                 : dit_int8_rowwise_gemm_bf16out_t128x128x64_w64x64x64_s4(
                       actI8, mWeightI8Device, actScale, mWeightScaleDevice,
                       outputs[0], static_cast<int32_t>(M), mN, K, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t FusedRmsNormLinearInt8Plugin::onShapeChange(PluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* FusedRmsNormLinearInt8Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* FusedRmsNormLinearInt8Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    if (!mWeightI4Host.empty()) {
        mDataToSerialize.emplace_back(
            PluginField("weight_i4", mWeightI4Host.data(), PluginFieldType::kINT8,
                        static_cast<int32_t>(mWeightI4Host.size())));
    } else {
        mDataToSerialize.emplace_back(
            PluginField("weight_i8", mWeightI8Host.data(), PluginFieldType::kINT8,
                        static_cast<int32_t>(mWeightI8Host.size())));
    }
    mDataToSerialize.emplace_back(
        PluginField("weight_scale", mWeightScaleHost.data(), PluginFieldType::kFLOAT32,
                    static_cast<int32_t>(mWeightScaleHost.size())));
    mDataToSerialize.emplace_back(
        PluginField("gamma", mGammaBf16Host.data(), PluginFieldType::kBF16,
                    static_cast<int32_t>(mGammaBf16Host.size())));
    if (!mStaticActScaleXHost.empty()) {
        mDataToSerialize.emplace_back(
            PluginField("static_act_scale_x", mStaticActScaleXHost.data(),
                        PluginFieldType::kFLOAT32,
                        static_cast<int32_t>(mStaticActScaleXHost.size())));
    }
    mDataToSerialize.emplace_back(PluginField("N", &mN, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1));
    // MUST be serialized: configurePlugin does not re-run on deserialize, and
    // the baked weights are already folded with W·Hᵀ.
    mDataToSerialize.emplace_back(
        PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// Creator
// ============================================================================

FusedRmsNormLinearInt8PluginCreator::FusedRmsNormLinearInt8PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* FusedRmsNormLinearInt8PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* FusedRmsNormLinearInt8PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedRmsNormLinearInt8PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* FusedRmsNormLinearInt8PluginCreator::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void FusedRmsNormLinearInt8PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* FusedRmsNormLinearInt8PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new FusedRmsNormLinearInt8Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterFusedRmsNormLinearInt8(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::FusedRmsNormLinearInt8PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initFusedRmsNormLinearInt8Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::FusedRmsNormLinearInt8PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
