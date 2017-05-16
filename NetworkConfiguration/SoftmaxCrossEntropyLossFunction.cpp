#include "SoftmaxCrossEntropyLossFunction.h"

namespace NetworkConfiguration {

SoftmaxCrossEntropyLossFunction::SoftmaxCrossEntropyLossFunction(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName) :
  LossFunction(scope, networkOutput, expectedOutputsPlaceholder), m_LossNode(scope, networkOutput, expectedOutputsPlaceholder)
{
  /// expected outputs should also be one hot encoded
  if (networkOutputShape.size() != 2 || expectedOutputShape.size() != 2)
  {
    throw std::logic_error("Both network output and expected outputs should be rank 2 tensors for using cross entropy loss.\n");
  }
  if (!shapesFormEqual(networkOutputShape, expectedOutputShape))
  {
    throw std::logic_error("Shape of expected outputs is not equal to the shape of input given to the loss function.\n");
  }
  using namespace tensorflow::ops;

  m_Loss = Mean(m_Scope.WithOpName(placeholderName), m_LossNode.loss, 0);
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LossCreator ctor = [](LossBaseParams & params) { return new SoftmaxCrossEntropyLossFunction(params);};
  bool dummy = LossFactory::instance().registerClass("SoftmaxCrossEntropyLossFunction", ctor);
}