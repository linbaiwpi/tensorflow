/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h"

#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace dummy_test {

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,
  kLegacyPie,  // Legacy path used by the PIE team and related clients.
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  bool compute_row_sums = false;
  // Only used for sparse hybrid fully connected kernels.
  bool ledger_initialized;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

// Dummy delegate kernel.
class DummyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit DummyDelegateKernel(const DummyDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    for (int i = 0; i < inputs_.size(); ++i) {
      switch (builtin_code_[i]) {
        case kTfLiteBuiltinAdd:
        case kTfLiteBuiltinSub:
          TF_LITE_ENSURE_EQ(
              context,
              myAddSub(context, builtin_code_[i], node, i),
              kTfLiteOk);

        case kTfLiteBuiltinFullyConnected:
          TF_LITE_ENSURE_EQ(
              context,
              dummyFullyConnected(context, node, i),
              kTfLiteOk);
      }
    }
    return kTfLiteOk;
  }

 private:
  TfLiteStatus dummyFullyConnected(TfLiteContext* context, TfLiteNode* node, int idx) {

    auto* params =
        reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
    const TfLiteTensor* filter;
    TF_LITE_ENSURE_OK(context,
                      GetInputSafe(context, node, kWeightsTensor, &filter));
    const TfLiteTensor* bias =
        (node->inputs->size == 3)
            ? GetOptionalInputTensor(context, node, kBiasTensor)
            : nullptr;
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context,
                      GetOutputSafe(context, node, kOutputTensor, &output));
    // Do nothing if expected output is empty.
    if (NumElements(output) == 0) {
      return kTfLiteOk;
    }

    switch (filter->type) {
      case kTfLiteFloat32:
        return dummyFullyConnectedFloat(context, node, params, data, input, filter,
                                        bias, output);
      default:
        context->ReportError(context,
                             "Filter data type %s currently not supported.",
                             TfLiteTypeGetName(filter->type));
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  TfLiteStatus dummyFullyConnectedFloat(TfLiteContext* context, TfLiteNode* node,
                                        TfLiteFullyConnectedParams* params, OpData* data,
                                        const TfLiteTensor* input, const TfLiteTensor* filter,
                                        const TfLiteTensor* bias, TfLiteTensor* output) {
    
    return kTfLiteOk;
  }

  TfLiteStatus myAddSub(TfLiteContext* context, int builtin_code, TfLiteNode* node, int idx) {
    // Get the node input tensors.
    // Add/Sub operation accepts 2 inputs.
    auto& input_tensor_1 = context->tensors[inputs_[idx][0]];
    auto& input_tensor_2 = context->tensors[inputs_[idx][1]];
    auto& output_tensor = context->tensors[outputs_[idx][0]];
    TF_LITE_ENSURE_EQ(
        context,
        ComputeResult(context, builtin_code_[idx], &input_tensor_1,
                      &input_tensor_2, &output_tensor),
        kTfLiteOk);
}

  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
        NumElements(input_tensor_1) != NumElements(output_tensor)) {
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);
    //std::cout << "===> " << builtin_code << std::endl;
    std::cout << "===> my_delegate.cc - Line 91" << std::endl;
    for (int i = 0; i < NumElements(input_tensor_1); ++i) {
      if (builtin_code == kTfLiteBuiltinAdd)
        output[i] = input_1[i] + input_2[i];
      else
        output[i] = input_1[i] - input_2[i];
    }
    return kTfLiteOk;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;

  const DummyDelegateOptions options_;
};

// DummyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class DummyDelegate : public SimpleDelegateInterface {
 public:
  explicit DummyDelegate(const DummyDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    switch (registration->builtin_code){
      case kTfLiteBuiltinAdd:
      case kTfLiteBuiltinFullyConnected:
      case kTfLiteBuiltinSub:
        return true;
      default:
        return false;
    }
    /*
    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32)
        return false;
    }
    return true;
    */
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "DummyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<DummyDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const DummyDelegateOptions options_;
};

}  // namespace dummy_test
}  // namespace tflite

DummyDelegateOptions TfLiteDummyDelegateOptionsDefault() {
  DummyDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options) {
  std::unique_ptr<tflite::dummy_test::DummyDelegate> dummy(
      new tflite::dummy_test::DummyDelegate(
          options ? *options : TfLiteDummyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(dummy));
}

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
