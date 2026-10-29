// IPluginV3 wrapper: per-row INT8 quant + INT8 Linear (no bias) + residual.
// Used for o_proj and down_proj sites (post-SDPA / post-silu·mul) in LLM L3+.

#include "plugin_field_util.h"
#include "int4_unpack_util.h"
#include "per_row_int8_linear_residual_plugin.h"
#include "dit_int8_rowwise_v2.h"          // per_row_quant
#include "dit_int8_rowwise_v2_fused.h"    // fused gemm_residual + t128 variant

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"PerRowInt8LinearResidual"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};

inline size_t workspaceBytes(int32_t M, int32_t K) {
    size_t a = (static_cast<size_t>(M) * K + 127) & ~static_cast<size_t>(127);
    size_t s = (static_cast<size_t>(M) * sizeof(float) + 127) & ~static_cast<size_t>(127);
    return a + s;
}
}  // anon

PluginFieldCollection PerRowInt8LinearResidualPluginCreator::mFieldCollection{};
std::vector<PluginField> PerRowInt8LinearResidualPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(PerRowInt8LinearResidualPluginCreator);

PerRowInt8LinearResidualPlugin::PerRowInt8LinearResidualPlugin(
    std::string const& name,
    std::vector<int8_t> weightI8,
    std::vector<float>  weightScale,
    int32_t N, int32_t K, int32_t rotBlockSize)
    : mLayerName(name)
    , mWeightI8Host(std::move(weightI8))
    , mWeightScaleHost(std::move(weightScale))
    , mN(N), mK(K), mRotBlockSize(rotBlockSize) {
    mNamespace = kPLUGIN_NAMESPACE;
}

PerRowInt8LinearResidualPlugin::PerRowInt8LinearResidualPlugin(
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
        } else if (n == "static_act_scale_x") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mStaticActScaleXHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        } else if (n == "N") {
            mN = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "K") {
            mK = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "rot_block_size") {
            mRotBlockSize = *static_cast<int32_t const*>(fc->fields[i].data);
        } else if (n == "act_scale_pre") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mScalePreHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        } else if (n == "act_scale_ch") {
            auto const* p = static_cast<float const*>(fc->fields[i].data);
            mScaleChHost.assign(p, p + gr00t::fieldElemCount(fc->fields[i], sizeof(float)));
        }
    }
}

PerRowInt8LinearResidualPlugin::~PerRowInt8LinearResidualPlugin() {
    if (mWeightI8Device)         cudaFree(mWeightI8Device);
    if (mWeightScaleDevice)      cudaFree(mWeightScaleDevice);
    if (mStaticActScaleXDevice)  cudaFree(mStaticActScaleXDevice);
    if (mScalePreDevice)         cudaFree(mScalePreDevice);
    if (mScaleChDevice)          cudaFree(mScaleChDevice);
}

void PerRowInt8LinearResidualPlugin::ensureWeightsOnDevice() {
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
    upload(&mScalePreDevice, mScalePreHost.data(), mScalePreHost.size() * sizeof(float));
    upload(&mScaleChDevice,  mScaleChHost.data(),  mScaleChHost.size()  * sizeof(float));
    upload(&mStaticActScaleXDevice, mStaticActScaleXHost.data(),
           mStaticActScaleXHost.size() * sizeof(float));
}

IPluginCapability* PerRowInt8LinearResidualPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* PerRowInt8LinearResidualPlugin::clone() noexcept {
    try {
        auto* p = new PerRowInt8LinearResidualPlugin(
            mLayerName, mWeightI8Host, mWeightScaleHost, mN, mK, mRotBlockSize);
        p->mStaticActScaleXHost = mStaticActScaleXHost;
        p->mWeightI4Host = mWeightI4Host;
        // clone() is what attachToContext hands the runtime; a member missing
        // here is gone at runtime, and a dropped SmoothQuant vector quantizes
        // un-smoothed with no symptom.
        p->mScalePreHost = mScalePreHost;
        p->mScaleChHost = mScaleChHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* PerRowInt8LinearResidualPlugin::getPluginName() const noexcept     { return kPLUGIN_NAME; }
char const* PerRowInt8LinearResidualPlugin::getPluginVersion() const noexcept  { return kPLUGIN_VERSION; }
char const* PerRowInt8LinearResidualPlugin::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void PerRowInt8LinearResidualPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t PerRowInt8LinearResidualPlugin::getNbOutputs() const noexcept { return 1; }

int32_t PerRowInt8LinearResidualPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t PerRowInt8LinearResidualPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    // Output shape = residual shape (= [B, S, N]).
    outputs[0].nbDims = inputs[1].nbDims;
    for (int32_t i = 0; i < inputs[1].nbDims - 1; ++i) {
        outputs[0].d[i] = inputs[1].d[i];
    }
    outputs[0].d[inputs[1].nbDims - 1] = exprBuilder.constant(mN);
    return 0;
}

bool PerRowInt8LinearResidualPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kBF16 && d.format == PluginFormat::kLINEAR;
}

int32_t PerRowInt8LinearResidualPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t PerRowInt8LinearResidualPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    int64_t Mmax = 1;
    for (int32_t i = 0; i < inputs[0].max.nbDims - 1; ++i) Mmax *= inputs[0].max.d[i];
    int32_t Kmax = inputs[0].max.d[inputs[0].max.nbDims - 1];
    return workspaceBytes(static_cast<int32_t>(Mmax), Kmax);
}

int32_t PerRowInt8LinearResidualPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();

        auto const& xDesc = inputDesc[0];
        int32_t nbDims = xDesc.dims.nbDims;
        int64_t M = 1;
        for (int32_t i = 0; i < nbDims - 1; ++i) M *= xDesc.dims.d[i];
        int32_t K = xDesc.dims.d[nbDims - 1];
        assert(K == mK);

        size_t aBytes = (static_cast<size_t>(M) * K + 127) & ~static_cast<size_t>(127);
        auto* ws = static_cast<uint8_t*>(workspace);
        int8_t* actI8    = reinterpret_cast<int8_t*>(ws);
        float*  actScale = reinterpret_cast<float*>(ws + aBytes);

        // Step 1: per-row INT8 quant.
        int rc;
        if (mStaticActScaleXDevice != nullptr &&
            static_cast<int>(mStaticActScaleXHost.size()) >= static_cast<int>(M)) {
            // Static scales are calibrated pre-rotation; pairing them with a
            // rotation would quantize rotated values against unrotated scales.
            // Fail loudly rather than emit a wrong engine.
            if (mRotBlockSize > 1) return -1;
            rc = dit_int8_per_row_static_quant_bf16(
                inputs[0], mStaticActScaleXDevice,
                actI8, actScale,
                static_cast<int32_t>(M), K, stream);
        } else {
            // One fold order or the other, never both: they divide the same
            // scale onto opposite axes of the rotation.
            if (!mScalePreHost.empty() && !mScaleChHost.empty()) return -6;
            rc = dit_int8_per_row_quant_fwht_bf16(
                inputs[0], mScalePreDevice, mScaleChDevice,
                actI8, actScale,
                static_cast<int32_t>(M), K, mRotBlockSize, stream);
        }
        if (rc != 0) return rc;

        // Step 2: fused INT8 GEMM + residual (no bias), t128x128x64 tile.
        // v4 ladder: full EVT fp32 precision (no bf16_add 2-step trade-off) +
        // tile-autotune winner. See test_llm_tile_autotune.py.
        // Tile choice, measured (RTX 4070 Ti Super, SM89): the 128x128x64 tile was
        // picked for the GR00T DiT and leaves the machine idle on the shapes the
        // action experts actually run. At M=10 it computes 128 rows for 10 real
        // ones and halves the CTA count; a projection with N=1024 gets 8 CTAs.
        // 64x64x64 wins 2.0-2.9x at M<=64 on every shape measured. Above that the
        // picture is shape-dependent and the LLM has its own autotune, so the
        // switch keys on M alone. Same kernel and EVT epilogue; launch shape only.
        bool const smallTile = (M <= 64);  // action-expert row counts only
        rc = smallTile
                 ? dit_int8_rowwise_gemm_residual_bf16out(
                       actI8, mWeightI8Device, actScale, mWeightScaleDevice,
                       inputs[1], outputs[0], static_cast<int32_t>(M), mN, K, stream)
                 : dit_int8_rowwise_gemm_residual_bf16out_t128x128x64_w64x64x64_s4(
                       actI8, mWeightI8Device, actScale, mWeightScaleDevice,
                       inputs[1], outputs[0], static_cast<int32_t>(M), mN, K, stream);
        return rc;
    } catch (...) { return -1; }
}

int32_t PerRowInt8LinearResidualPlugin::onShapeChange(PluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* PerRowInt8LinearResidualPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* PerRowInt8LinearResidualPlugin::getFieldsToSerialize() noexcept {
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
    if (!mStaticActScaleXHost.empty()) {
        mDataToSerialize.emplace_back(
            PluginField("static_act_scale_x", mStaticActScaleXHost.data(),
                        PluginFieldType::kFLOAT32,
                        static_cast<int32_t>(mStaticActScaleXHost.size())));
    }
    mDataToSerialize.emplace_back(PluginField("N", &mN, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    // MUST be serialized; see mRotBlockSize in the header.
    mDataToSerialize.emplace_back(
        PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre", mScalePreHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreHost.size())));
    mDataToSerialize.emplace_back(PluginField("act_scale_ch", mScaleChHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScaleChHost.size())));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// Creator
// ============================================================================

PerRowInt8LinearResidualPluginCreator::PerRowInt8LinearResidualPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_i8", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("static_act_scale_x", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("act_scale_ch", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* PerRowInt8LinearResidualPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* PerRowInt8LinearResidualPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* PerRowInt8LinearResidualPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* PerRowInt8LinearResidualPluginCreator::getPluginNamespace() const noexcept{ return mNamespace.c_str(); }
void PerRowInt8LinearResidualPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* PerRowInt8LinearResidualPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new PerRowInt8LinearResidualPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterPerRowInt8LinearResidual(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::PerRowInt8LinearResidualPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initPerRowInt8LinearResidualPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::PerRowInt8LinearResidualPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
