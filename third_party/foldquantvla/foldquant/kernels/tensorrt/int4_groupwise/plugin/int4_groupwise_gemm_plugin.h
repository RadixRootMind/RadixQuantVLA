/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
/*!
 * @brief TensorRT plugin for INT4 group-wise quantized GEMM
 *
 * Implements efficient INT4 quantized matrix multiplication with group-wise quantization.
 * Used for quantized weight matrix multiplications in LLM inference.
 */
class Int4GroupwiseGemmPlugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime
{
public:
    /*!
     * @brief Construct INT4 group-wise GEMM plugin
     * @param name Layer name
     * @param N Output dimension (columns in weight matrix)
     * @param K Input dimension (rows in weight matrix)
     * @param groupSize Quantization group size
     */
    Int4GroupwiseGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize);

    /*!
     * @brief Construct from field collection
     * @param name Layer name
     * @param fc Plugin field collection
     */
    Int4GroupwiseGemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    //! @brief Deleted default constructor
    Int4GroupwiseGemmPlugin() = delete;

    //! @brief Deleted copy constructor
    Int4GroupwiseGemmPlugin(Int4GroupwiseGemmPlugin const&) = delete;

    //! @brief Destructor
    ~Int4GroupwiseGemmPlugin() override;

    //! @brief Return the plugin capability interface for given type
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    //! @brief Clone the plugin for use in another network
    //! @return Cloned plugin instance
    nvinfer1::IPluginV3* clone() noexcept override;

    //! @brief Get plugin name
    //! @return Plugin name string
    char const* getPluginName() const noexcept override;

    //! @brief Get plugin version
    //! @return Version string
    char const* getPluginVersion() const noexcept override;

    //! @brief Get plugin namespace
    //! @return Namespace string
    char const* getPluginNamespace() const noexcept override;

    //! @brief Get number of output tensors
    //! @return Number of outputs (1)
    int32_t getNbOutputs() const noexcept override;

    //! @brief Get output tensor data types
    //! @param outputTypes Output array for data types
    //! @param nbOutputs Number of outputs
    //! @param inputTypes Input data types (not used)
    //! @param nbInputs Number of inputs (not used)
    //! @return 0 on success, non-zero on error
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* /* inputTypes */, int32_t /* nbInputs */) const noexcept override;

    //! @brief Get output tensor shapes
    //! @param inputs Input dimensions
    //! @param nbInputs Number of inputs
    //! @param shapeInputs Shape tensor inputs (not used)
    //! @param nbShapeInputs Number of shape inputs (not used)
    //! @param outputs Output dimensions
    //! @param nbOutputs Number of outputs
    //! @param exprBuilder Expression builder for dynamic shapes
    //! @return 0 on success, non-zero on error
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    //! @brief Check if format combination is supported
    //! @param pos Position in input/output array
    //! @param inOut Input and output tensor descriptors
    //! @param nbInputs Number of inputs
    //! @param nbOutputs Number of outputs
    //! @return True if supported
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    //! @brief Configure plugin with tensor descriptions
    //! @param in Input tensor descriptors (not used)
    //! @param nbInputs Number of inputs (not used)
    //! @param out Output tensor descriptors (not used)
    //! @param nbOutputs Number of outputs (not used)
    //! @details This operation is a no-op for this plugin type.
    //! @return 0 on success, non-zero on error
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* /* in */, int32_t /* nbInputs */,
        nvinfer1::DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept override;

    //! @brief Get workspace size required for execution
    //! @param inputs Input tensor descriptors (not used)
    //! @param nbInputs Number of inputs (not used)
    //! @param outputs Output tensor descriptors (not used)
    //! @param nbOutputs Number of outputs (not used)
    //! @return Zero bytes (this plugin does not use workspace)
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
        nvinfer1::DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept override;

    //! @brief Execute the plugin
    //! @param inputDesc Input tensor descriptors
    //! @param outputDesc Output tensor descriptors
    //! @param inputs Input tensor pointers
    //! @param outputs Output tensor pointers
    //! @param workspace Workspace pointer
    //! @param stream CUDA stream
    //! @return 0 on success, non-zero on error
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* /* outputDesc */,
        void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept override;

    //! @brief Called when input/output shapes change during runtime
    //! @param in Input tensor descriptors (unused)
    //! @param nbInputs Number of inputs (unused)
    //! @param out Output tensor descriptors (unused)
    //! @param nbOutputs Number of outputs (unused)
    //! @details This is a no-op for this plugin type.
    //! @return 0 to signal success
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
        nvinfer1::PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept override;

    //! @brief Attach plugin to an execution context
    //! @param context Plugin resource context (not used by this plugin)
    //! @return Cloned plugin attached to context
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* /* context */) noexcept override;

    //! @brief Get plugin fields for serialization
    //! @return Field collection for serialization
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    //! @brief Set plugin namespace
    //! @param pluginNamespace Namespace string
    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mGemmN{};
    int32_t mGemmK{};
    int32_t mGroupSize{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/*!
 * @brief Factory for creating Int4GroupwiseGemmPlugin instances
 *
 * Handles plugin registration and creation in TensorRT.
 */
class Int4GroupwiseGemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    //! @brief Constructor
    Int4GroupwiseGemmPluginCreator();

    //! @brief Destructor
    ~Int4GroupwiseGemmPluginCreator() override = default;

    //! @brief Get plugin name
    //! @return Plugin name string
    char const* getPluginName() const noexcept override;

    //! @brief Get plugin version
    //! @return Version string
    char const* getPluginVersion() const noexcept override;

    //! @brief Get plugin field names
    //! @return Field collection
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    //! @brief Get plugin namespace
    //! @return Namespace string
    char const* getPluginNamespace() const noexcept override;

    //! @brief Set plugin namespace
    //! @param pluginNamespace Namespace string
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! @brief Create plugin from field collection
    //! @param name Plugin name
    //! @param fc Field collection with parameters
    //! @param phase TensorRT phase (build or runtime) - not used by this plugin
    //! @return Created plugin instance
    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase /* phase */) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
