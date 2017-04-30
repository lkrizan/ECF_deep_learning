#include "MeanSquaredLossFunction.h"

namespace NetworkConfiguration {

MeanSquaredLossFunction::MeanSquaredLossFunction(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName) : LossFunction(scope)
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

}

}	// namespace NetworkConfiguration

  // register class in factory
namespace {
  using namespace NetworkConfiguration;
  LossCreator ctor = [](LossFunction::LossBaseParams & params) { return new MeanSquaredLossFunction(params);};
  bool dummy = LossFactory::instance().registerClass("MeanSquaredLossFunction", ctor);
}

