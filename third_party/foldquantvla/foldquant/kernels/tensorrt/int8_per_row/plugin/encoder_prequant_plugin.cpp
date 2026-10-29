// EncoderPreQuant: runs the per-row INT8 quant of the cross-attn encoder
// once per forward; emits (int8_enc, scale_enc) shared by all 16 downstream
// FusedAdaLnQuantCrossAttnPrequantized / FusedCrossAttnFull blocks.

#include "plugin_field_util.h"
#include "encoder_prequant_plugin.h"
#include "dit_int8_rowwise_v2.h"  // dit_int8_per_row_{quant,static_quant}_bf16

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"EncoderPreQuant"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};
}  // anon

PluginFieldCollection EncoderPreQuantPluginCreator::mFieldCollection{};
std::vector<PluginField> EncoderPreQuantPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EncoderPreQuantPluginCreator);

EncoderPreQuantPlugin::EncoderPreQuantPlugin(std::string const& name, int32_t KEnc)
    : mLayerName(name), mKEnc(KEnc) {
    mNamespace = kPLUGIN_NAMESPACE;
}

EncoderPreQuantPlugin::EncoderPreQuantPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "static_act_scale_enc") {
            auto const* p = static_cast<float const*>(f.data);
            mStaticActScaleEncHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float)));
        } else if (n == "K_enc") {
            mKEnc = *static_cast<int32_t const*>(f.data);
        } else if (n == "rot_block_size") {
            mRotBlockSize = *static_cast<int32_t const*>(f.data);
        } else if (n == "act_scale_pre_enc") {
            auto const* p = static_cast<float const*>(f.data);
            mScalePreEncHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float)));
        }
    }
}

EncoderPreQuantPlugin::~EncoderPreQuantPlugin() {
    if (mStaticActScaleEncDevice) cudaFree(mStaticActScaleEncDevice);
    if (mScalePreEncDevice) cudaFree(mScalePreEncDevice);
}

void EncoderPreQuantPlugin::ensureWeightsOnDevice() {
    if (!mScalePreEncDevice && !mScalePreEncHost.empty()) {
        size_t bytes = mScalePreEncHost.size() * sizeof(float);
        if (cudaMalloc(&mScalePreEncDevice, bytes) == cudaSuccess) {
            cudaMemcpy(mScalePreEncDevice, mScalePreEncHost.data(), bytes, cudaMemcpyHostToDevice);
        }
    }
    if (!mStaticActScaleEncDevice && !mStaticActScaleEncHost.empty()) {
        size_t bytes = mStaticActScaleEncHost.size() * sizeof(float);
        cudaMalloc(&mStaticActScaleEncDevice, bytes);
        cudaMemcpy(mStaticActScaleEncDevice, mStaticActScaleEncHost.data(), bytes, cudaMemcpyHostToDevice);
    }
}

IPluginCapability* EncoderPreQuantPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* EncoderPreQuantPlugin::clone() noexcept {
    try {
        auto* p = new EncoderPreQuantPlugin(mLayerName, mKEnc);
        p->mStaticActScaleEncHost = mStaticActScaleEncHost;
        // The clone is the object TensorRT actually runs. A member left behind
        // here is not a crash; it is a plugin that quietly stops rotating.
        p->mRotBlockSize = mRotBlockSize;
        p->mScalePreEncHost = mScalePreEncHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* EncoderPreQuantPlugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* EncoderPreQuantPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* EncoderPreQuantPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void EncoderPreQuantPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t EncoderPreQuantPlugin::getNbOutputs() const noexcept { return 2; }

int32_t EncoderPreQuantPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 2);
    // INT8 payload is declared as INT32 packed (4 INT8 bytes per INT32 element)
    // to bypass TRT 10.3's INT8 BuilderFlag requirement; without this the
    // builder cascades dynamic-range requirements onto every BF16 tensor in
    // the graph (~1.5× E2E slowdown).
    outputTypes[0] = DataType::kINT32;   // encoder_i8 packed (B, S_enc, K_enc/4)
    outputTypes[1] = DataType::kFLOAT;   // encoder_scale: (B, S_enc)
    return 0;
}

int32_t EncoderPreQuantPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
    assert(nbInputs == 1 && nbOutputs == 2);
    // output 0: encoder packed (B, S_enc, K_enc / 4); K_enc is static (=mKEnc).
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[0].d[i] = inputs[0].d[i];
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mKEnc / 4);
    // output 1: (B, S_enc) per-row scale, drop last dim
    outputs[1].nbDims = inputs[0].nbDims - 1;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[1].d[i] = inputs[0].d[i];
    return 0;
}

bool EncoderPreQuantPlugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 2);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    if (pos == 0) return d.type == DataType::kBF16;     // input encoder
    if (pos == 1) return d.type == DataType::kINT32;    // packed encoder_i8 (4 INT8/INT32)
    if (pos == 2) return d.type == DataType::kFLOAT;    // output scale
    return false;
}

int32_t EncoderPreQuantPlugin::configurePlugin(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t EncoderPreQuantPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    return 0;
}

int32_t EncoderPreQuantPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* /*workspace*/, cudaStream_t stream) noexcept {
    try {
        ensureWeightsOnDevice();

        auto const& encDesc = inputDesc[0];
        int64_t MEnc = 1;
        for (int32_t i = 0; i < encDesc.dims.nbDims - 1; ++i) MEnc *= encDesc.dims.d[i];
        int32_t KEnc = encDesc.dims.d[encDesc.dims.nbDims - 1];

        int rc;
        if (mRotBlockSize > 0) {
            // FoldQuant: rotate then quantize, in one launch. The amax is taken on
            // the rotated row inside the kernel; measuring it before the
            // rotation would scale every channel from the wrong basis.
            rc = dit_int8_per_row_quant_fwht_bf16(
                inputs[0], mScalePreEncDevice, /*act_scale_ch=*/nullptr,
                outputs[0], outputs[1],
                static_cast<int32_t>(MEnc), KEnc, mRotBlockSize, stream);
        } else if (mStaticActScaleEncDevice != nullptr &&
            (int)mStaticActScaleEncHost.size() >= static_cast<int>(MEnc)) {
            rc = dit_int8_per_row_static_quant_bf16(
                inputs[0], mStaticActScaleEncDevice,
                outputs[0], outputs[1],
                static_cast<int32_t>(MEnc), KEnc, stream);
        } else {
            rc = dit_int8_per_row_quant_bf16(
                inputs[0], outputs[0], outputs[1],
                static_cast<int32_t>(MEnc), KEnc, stream);
        }
        return rc;
    } catch (...) { return -1; }
}

int32_t EncoderPreQuantPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* EncoderPreQuantPlugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    return clone();
}

PluginFieldCollection const* EncoderPreQuantPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    if (!mStaticActScaleEncHost.empty()) {
        mDataToSerialize.emplace_back(PluginField("static_act_scale_enc",
            mStaticActScaleEncHost.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(mStaticActScaleEncHost.size())));
    }
    mDataToSerialize.emplace_back(PluginField("K_enc", &mKEnc, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    if (!mScalePreEncHost.empty()) {
        mDataToSerialize.emplace_back(PluginField("act_scale_pre_enc", mScalePreEncHost.data(),
            PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreEncHost.size())));
    }
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

EncoderPreQuantPluginCreator::EncoderPreQuantPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("static_act_scale_enc", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("K_enc", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_enc", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* EncoderPreQuantPluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* EncoderPreQuantPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* EncoderPreQuantPluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* EncoderPreQuantPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void EncoderPreQuantPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* EncoderPreQuantPluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new EncoderPreQuantPlugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterEncoderPreQuant(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::EncoderPreQuantPluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initEncoderPreQuantPlugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::EncoderPreQuantPluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
