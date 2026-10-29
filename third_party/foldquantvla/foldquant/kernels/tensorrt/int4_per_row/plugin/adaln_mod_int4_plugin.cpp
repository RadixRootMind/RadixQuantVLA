// AdaLNModInt4: INT4 weight-only (W4A16) AdaLN modulation linear.
// See adaln_mod_int4_plugin.h. Skeleton cloned from encoder_prequant_int4_plugin.cpp.

#include "plugin_field_util.h"
#include "adaln_mod_int4_plugin.h"
#include "dit_int4_rowwise.h"

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;

namespace gr00t {
namespace v1 {
namespace plugins {

namespace {
constexpr char const* kPLUGIN_NAME{"AdaLNModInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};
}  // anon

PluginFieldCollection AdaLNModInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> AdaLNModInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AdaLNModInt4PluginCreator);

AdaLNModInt4Plugin::AdaLNModInt4Plugin(std::string const& name,
    std::vector<int8_t> wI4, std::vector<uint16_t> wScale, std::vector<uint16_t> bias,
    int32_t inDim, int32_t outDim, int32_t actBits)
    : mLayerName(name), mWI4Host(std::move(wI4)), mWScaleHost(std::move(wScale))
    , mBiasHost(std::move(bias)), mInDim(inDim), mOutDim(outDim), mActBits(actBits) {
    mNamespace = kPLUGIN_NAMESPACE;
}

AdaLNModInt4Plugin::AdaLNModInt4Plugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "weight_i4")          { auto* p = static_cast<int8_t const*>(f.data);   mWI4Host.assign(p, p + gr00t::fieldElemCount(f, 1)); }
        else if (n == "weight_scale")  { auto* p = static_cast<uint16_t const*>(f.data); mWScaleHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "bias")          { auto* p = static_cast<uint16_t const*>(f.data); mBiasHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(uint16_t))); }
        else if (n == "in_dim")        { mInDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "out_dim")       { mOutDim = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_bits")      { mActBits = *static_cast<int32_t const*>(f.data); }
    }
}

AdaLNModInt4Plugin::~AdaLNModInt4Plugin() {
    // mXDevice are non-owning views into the shared resource; release our ref.
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights().
enum AdaLNW { W_I4 = 0, W_SCALE, W_BIAS, W_COUNT };
}  // anon

std::vector<WeightSpec> AdaLNModInt4Plugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_I4]    = {mWI4Host.data(),    mWI4Host.size()};
    s[W_SCALE] = {mWScaleHost.data(), mWScaleHost.size() * sizeof(uint16_t)};
    s[W_BIAS]  = {mBiasHost.data(),   mBiasHost.size()   * sizeof(uint16_t)};
    return s;
}

void AdaLNModInt4Plugin::bindDeviceWeights() {
    mWI4Device    = mShared->buf(W_I4);
    mWScaleDevice = mShared->buf(W_SCALE);
    mBiasDevice   = mShared->buf(W_BIAS);
}

IPluginCapability* AdaLNModInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* AdaLNModInt4Plugin::clone() noexcept {
    try {
        auto* p = new AdaLNModInt4Plugin(mLayerName, mWI4Host, mWScaleHost, mBiasHost, mInDim, mOutDim, mActBits);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* AdaLNModInt4Plugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* AdaLNModInt4Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* AdaLNModInt4Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void AdaLNModInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t AdaLNModInt4Plugin::getNbOutputs() const noexcept { return 1; }

int32_t AdaLNModInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t /*nbOutputs*/,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t AdaLNModInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t /*nbInputs*/,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t /*nbOutputs*/, IExprBuilder& exprBuilder) noexcept {
    // Same leading dims as the input; last dim → out_dim.
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[0].d[i] = inputs[0].d[i];
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mOutDim);
    return 0;
}

bool AdaLNModInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1);
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    return d.type == DataType::kBF16;   // input (pos 0) and output (pos 1) both BF16
}

int32_t AdaLNModInt4Plugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t AdaLNModInt4Plugin::getWorkspaceSize(DynamicPluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    return 0;
}

int32_t AdaLNModInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* /*workspace*/, cudaStream_t stream) noexcept {
    try {
        if (mShared == nullptr) return -1;  // weights bound in attachToContext
        auto const& xDesc = inputDesc[0];
        int64_t M = 1;
        for (int32_t i = 0; i < xDesc.dims.nbDims - 1; ++i) M *= xDesc.dims.d[i];
        auto fn = (mActBits == 4) ? dit_adaln_gemv_int4a4_bf16
                                  : dit_adaln_gemv_int4_bf16;
        return fn(
            inputs[0], mWI4Device, mWScaleDevice, mBiasDevice, outputs[0],
            static_cast<int32_t>(M), mInDim, mOutDim, stream);
    } catch (...) { return -1; }
}

int32_t AdaLNModInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* AdaLNModInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<AdaLNModInt4Plugin*>(clone());
        if (p == nullptr) return nullptr;
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

PluginFieldCollection const* AdaLNModInt4Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("weight_i4", mWI4Host.data(),
        PluginFieldType::kINT8, static_cast<int32_t>(mWI4Host.size())));
    mDataToSerialize.emplace_back(PluginField("weight_scale", mWScaleHost.data(),
        PluginFieldType::kBF16, static_cast<int32_t>(mWScaleHost.size())));
    mDataToSerialize.emplace_back(PluginField("bias", mBiasHost.data(),
        PluginFieldType::kBF16, static_cast<int32_t>(mBiasHost.size())));
    mDataToSerialize.emplace_back(PluginField("in_dim", &mInDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("out_dim", &mOutDim, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_bits", &mActBits, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

AdaLNModInt4PluginCreator::AdaLNModInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("weight_i4", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("weight_scale", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kBF16, 0));
    mPluginAttributes.emplace_back(PluginField("in_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("out_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_bits", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* AdaLNModInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* AdaLNModInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* AdaLNModInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* AdaLNModInt4PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void AdaLNModInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* AdaLNModInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new AdaLNModInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterAdaLNModInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::AdaLNModInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initAdaLNModInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::AdaLNModInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
