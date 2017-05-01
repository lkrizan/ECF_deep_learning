#include "CrossEntropyLossFunction.h"

namespace NetworkConfiguration {

CrossEntropyLossFunction::CrossEntropyLossFunction(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName) :
  LossFunction(scope)
{
  /// expected outputs should also be one hot encoded
  /// should be used only after softmax
  if (networkOutputShape.size() != 2 || expectedOutputShape.size() != 2)
  {
    throw std::logic_error("Both network output and expected outputs should be rank 2 tensors for using cross entropy loss.\n");
  }
  if (!shapesFormEqual(networkOutputShape, expectedOutputShape))
  {
    throw std::logic_error("Shape of expected outputs is not equal to the shape of input given to the loss function.\n");
  }
  using namespace tensorflow::ops;

  // only probablity relevant to the actual class are left in the matrix
  auto tempMultiply = Multiply(m_Scope, networkOutput, expectedOutputsPlaceholder);
  // now take those values
  auto tempMax = Max(m_Scope, tempMultiply, 1);
  // apply log function
  auto tempLog = Log(m_Scope, tempMax);
  // multiply by -1
  auto multiplyConstant = Const(m_Scope, 1);
  auto totalError = Multiply(m_Scope, tempLog, multiplyConstant);
  // take mean value
  m_Loss = Mean(scope.WithOpName(placeholderName), totalError, 0);
  
}

}   // namespace NetworkConfiguration

// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LossCreator ctor = [](LossBaseParams & params) { return new CrossEntropyLossFunction(params);};
  bool dummy = LossFactory::instance().registerClass("CrossEntropyLossFunction", ctor);
}
