#include "SoftmaxActivation.h"

namespace NetworkConfiguration {

SoftmaxActivation::SoftmaxActivation(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) :
  NonParameterizedLayer(scope, previousLayerOutput)
{
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Softmax(scope.WithOpName(outputName), m_Input);
}

tensorflow::Output SoftmaxActivation::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  throw std::logic_error("Error: Backward pass is not implemented for softmax activation. Use SoftmaxCrossEntropyLoss.\n");
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) { return new SoftmaxActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("SoftmaxActivation", ctor);
}

