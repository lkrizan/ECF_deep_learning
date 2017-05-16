#include "MeanSquaredLossFunction.h"

namespace NetworkConfiguration {

MeanSquaredLossFunction::MeanSquaredLossFunction(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName) : 
  LossFunction(scope, networkOutput, expectedOutputsPlaceholder)
{
  if (!shapesFormEqual(networkOutputShape, expectedOutputShape))
  {
    throw std::logic_error("Shape of expected outputs is not equal to the shape of input given to the loss function.\n");
  }

  // create graph nodes
  using namespace tensorflow::ops;
  auto diff = Subtract(m_Scope, m_NetworkOutput, m_ExpectedOutputs);
  auto squaredDiff = Square(m_Scope, diff);
  auto exampleMean = Mean(m_Scope, squaredDiff, 1);
  m_Loss = Mean(m_Scope.WithOpName(placeholderName), exampleMean, 0);

}

tensorflow::Output MeanSquaredLossFunction::backward()
{
  using namespace tensorflow::ops;
  return Subtract(m_Scope, m_NetworkOutput, m_ExpectedOutputs);
}

}	// namespace NetworkConfiguration

  // register class in factory
namespace {
  using namespace NetworkConfiguration;
  LossCreator ctor = [](LossBaseParams & params) { return new MeanSquaredLossFunction(params);};
  bool dummy = LossFactory::instance().registerClass("MeanSquaredLossFunction", ctor);
}

