#include "MeanSquaredLossFunction.h"

namespace NetworkConfiguration {

NetworkConfiguration::MeanSquaredLossFunction::MeanSquaredLossFunction(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName)
{
  if (!shapesFormEqual(networkOutputShape, expectedOutputShape))
  {
    throw std::logic_error("Shape of expected outputs is not equal to the shape of input given to the loss function.\n");
  }

  // create graph nodes
  using namespace tensorflow::ops;
  auto diff = Subtract(scope, networkOutput, expectedOutputsPlaceholder);
  auto squaredDiff = Square(scope, diff);
  m_Loss = Mean(scope.WithOpName(placeholderName), squaredDiff, 0);

  // set output shape
  m_OutputShape.push_back(1);
}

const tensorflow::Output & MeanSquaredLossFunction::getLossOutput() const
{
  return m_Loss;
}

Shape MeanSquaredLossFunction::outputShape() const
{
  return m_OutputShape;
}

}	// namespace NetworkConfiguration
