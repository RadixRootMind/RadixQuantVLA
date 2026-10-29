// EncoderPreQuantInt4: shared-rotation + per-row INT4 quant of the encoder, once.
// Clone of encoder_prequant_plugin.cpp.

#include "plugin_field_util.h"
#include "encoder_prequant_int4_plugin.h"
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
constexpr char const* kPLUGIN_NAME{"EncoderPreQuantInt4"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAMESPACE{"gr00t::v1"};
}  // anon

PluginFieldCollection EncoderPreQuantInt4PluginCreator::mFieldCollection{};
std::vector<PluginField> EncoderPreQuantInt4PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EncoderPreQuantInt4PluginCreator);

EncoderPreQuantInt4Plugin::EncoderPreQuantInt4Plugin(std::string const& name,
    std::vector<int32_t> permEnc, std::vector<float> rotEnc, int32_t KEnc, int32_t blockSize)
    : mLayerName(name), mPermEncHost(std::move(permEnc)), mRotEncHost(std::move(rotEnc))
    , mKEnc(KEnc), mBlockSize(blockSize) {
    mNamespace = kPLUGIN_NAMESPACE;
}

EncoderPreQuantInt4Plugin::EncoderPreQuantInt4Plugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name) {
    mNamespace = kPLUGIN_NAMESPACE;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        std::string n(fc->fields[i].name);
        auto const& f = fc->fields[i];
        if (n == "perm_enc")        { auto* p = static_cast<int32_t const*>(f.data); mPermEncHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(int32_t))); }
        else if (n == "rotation_enc"){auto* p = static_cast<float const*>(f.data); mRotEncHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
        else if (n == "K_enc")      { mKEnc = *static_cast<int32_t const*>(f.data); }
        else if (n == "block_size") { mBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "rot_block_size")    { mRotBlockSize = *static_cast<int32_t const*>(f.data); }
        else if (n == "act_scale_pre_enc") { auto* p = static_cast<float const*>(f.data); mScalePreEncHost.assign(p, p + gr00t::fieldElemCount(f, sizeof(float))); }
    }
}

EncoderPreQuantInt4Plugin::~EncoderPreQuantInt4Plugin() {
    // mXDevice are non-owning views into the shared resource; release our ref.
    if (!mResourceKey.empty()) releaseSharedWeights(mResourceKey);
}

namespace {
// Canonical order shared by hostWeightSpecs() and bindDeviceWeights().
enum EncoderW { W_PERM_ENC = 0, W_ROT_ENC, W_SCALE_PRE_ENC, W_COUNT };
}  // anon

std::vector<WeightSpec> EncoderPreQuantInt4Plugin::hostWeightSpecs() const {
    std::vector<WeightSpec> s(W_COUNT);
    s[W_PERM_ENC] = {mPermEncHost.data(), mPermEncHost.size() * sizeof(int32_t)};
    s[W_ROT_ENC]  = {mRotEncHost.data(),  mRotEncHost.size()  * sizeof(float)};
    s[W_SCALE_PRE_ENC] = {mScalePreEncHost.data(), mScalePreEncHost.size() * sizeof(float)};
    return s;
}

void EncoderPreQuantInt4Plugin::bindDeviceWeights() {
    mPermEncDevice = mShared->buf(W_PERM_ENC);
    mRotEncDevice  = mShared->buf(W_ROT_ENC);
    mScalePreEncDevice = mScalePreEncHost.empty() ? nullptr : mShared->buf(W_SCALE_PRE_ENC);
}

IPluginCapability* EncoderPreQuantInt4Plugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD)   return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (...) { return nullptr; }
}

IPluginV3* EncoderPreQuantInt4Plugin::clone() noexcept {
    try {
        auto* p = new EncoderPreQuantInt4Plugin(mLayerName, mPermEncHost, mRotEncHost, mKEnc, mBlockSize);
        // attachToContext() clones; a member missing here is gone at runtime.
        p->mRotBlockSize = mRotBlockSize;
        p->mScalePreEncHost = mScalePreEncHost;
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

char const* EncoderPreQuantInt4Plugin::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* EncoderPreQuantInt4Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* EncoderPreQuantInt4Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void EncoderPreQuantInt4Plugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

int32_t EncoderPreQuantInt4Plugin::getNbOutputs() const noexcept { return 2; }

int32_t EncoderPreQuantInt4Plugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept {
    assert(nbOutputs == 2);
    outputTypes[0] = DataType::kINT32;   // int4 packed (B, S_enc, K_enc/8)
    outputTypes[1] = DataType::kFLOAT;   // encoder_scale (B, S_enc)
    return 0;
}

int32_t EncoderPreQuantInt4Plugin::getOutputShapes(DimsExprs const* inputs, int32_t /*nbInputs*/,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t /*nbOutputs*/, IExprBuilder& exprBuilder) noexcept {
    // output 0: int4-packed (B, S_enc, K_enc/8 int32), K_enc/2 bytes per row.
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[0].d[i] = inputs[0].d[i];
    outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mKEnc / 8);
    // output 1: (B, S_enc) per-row scale.
    outputs[1].nbDims = inputs[0].nbDims - 1;
    for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i) outputs[1].d[i] = inputs[0].d[i];
    return 0;
}

bool EncoderPreQuantInt4Plugin::supportsFormatCombination(int32_t pos,
    DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 2);
    assert(pos < (nbInputs + nbOutputs));
    auto const& d = inOut[pos].desc;
    if (d.format != PluginFormat::kLINEAR) return false;
    if (pos == 0) return d.type == DataType::kBF16;
    if (pos == 1) return d.type == DataType::kINT32;
    if (pos == 2) return d.type == DataType::kFLOAT;
    return false;
}

int32_t EncoderPreQuantInt4Plugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept {
    return 0;
}

size_t EncoderPreQuantInt4Plugin::getWorkspaceSize(DynamicPluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    return 0;
}

int32_t EncoderPreQuantInt4Plugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs,
    void* const* outputs, void* /*workspace*/, cudaStream_t stream) noexcept {
    try {
        if (mShared == nullptr) return -1;  // weights bound in attachToContext
        auto const& encDesc = inputDesc[0];
        int64_t MEnc = 1;
        for (int32_t i = 0; i < encDesc.dims.nbDims - 1; ++i) MEnc *= encDesc.dims.d[i];
        int32_t KEnc = encDesc.dims.d[encDesc.dims.nbDims - 1];
        if (mRotBlockSize > 0) {
            // Butterfly: one kernel, no matrix read. The dense kernel this
            // replaces launches grid(MEnc) blocks and walks the rotation out of
            // global memory twice per row; it is the single most expensive node
            // in the DiT profile.
            return dit_int4_per_row_quant_fwht_bf16(
                inputs[0], mScalePreEncDevice, /*act_scale_ch=*/nullptr,
                outputs[0], outputs[1],
                static_cast<int32_t>(MEnc), KEnc, mRotBlockSize, /*act_clip=*/1.0f, stream);
        }
        return dit_int4_per_row_rotate_quant_bf16(
            inputs[0], mPermEncDevice, mRotEncDevice, outputs[0], outputs[1],
            static_cast<int32_t>(MEnc), KEnc, mBlockSize, stream);
    } catch (...) { return -1; }
}

int32_t EncoderPreQuantInt4Plugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept { return 0; }

IPluginV3* EncoderPreQuantInt4Plugin::attachToContext(IPluginResourceContext* /*ctx*/) noexcept {
    try {
        auto* p = static_cast<EncoderPreQuantInt4Plugin*>(clone());
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

PluginFieldCollection const* EncoderPreQuantInt4Plugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("perm_enc", mPermEncHost.data(),
        PluginFieldType::kINT32, static_cast<int32_t>(mPermEncHost.size())));
    mDataToSerialize.emplace_back(PluginField("rotation_enc", mRotEncHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mRotEncHost.size())));
    mDataToSerialize.emplace_back(PluginField("K_enc", &mKEnc, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("block_size", &mBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("rot_block_size", &mRotBlockSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("act_scale_pre_enc", mScalePreEncHost.data(),
        PluginFieldType::kFLOAT32, static_cast<int32_t>(mScalePreEncHost.size())));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

EncoderPreQuantInt4PluginCreator::EncoderPreQuantInt4PluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("perm_enc", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotation_enc", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("K_enc", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rot_block_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("act_scale_pre_enc", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
    mNamespace = kPLUGIN_NAMESPACE;
}

char const* EncoderPreQuantInt4PluginCreator::getPluginName() const noexcept    { return kPLUGIN_NAME; }
char const* EncoderPreQuantInt4PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* EncoderPreQuantInt4PluginCreator::getFieldNames() noexcept { return &mFieldCollection; }
char const* EncoderPreQuantInt4PluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void EncoderPreQuantInt4PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

IPluginV3* EncoderPreQuantInt4PluginCreator::createPlugin(char const* name,
    PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept {
    try {
        auto* p = new EncoderPreQuantInt4Plugin(std::string(name), fc);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (...) { return nullptr; }
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t

__attribute__((constructor))
static void _autoRegisterEncoderPreQuantInt4(void) {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return;
    static gr00t::v1::plugins::EncoderPreQuantInt4PluginCreator creator;
    registry->registerCreator(creator, "gr00t::v1");
}

extern "C" int initEncoderPreQuantInt4Plugin() {
    auto* registry = getPluginRegistry();
    if (registry == nullptr) return -1;
    static gr00t::v1::plugins::EncoderPreQuantInt4PluginCreator creator;
    bool ok = registry->registerCreator(creator, "gr00t::v1");
    return ok ? 0 : 1;
}
