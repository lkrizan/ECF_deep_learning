#include "ReLUActivation.h"

namespace NetworkConfiguration {

ReLUActivation::ReLUActivation(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) : 
  NonParameterizedLayer(scope, previousLayerOutput)
{
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Relu(scope.WithOpName(outputName), m_Input);
}

tensorflow::Output ReLUActivation::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  // gradient should be zero where layer inputs were <= 0
  using namespace tensorflow::ops;
  auto greaterThanZero = Greater(m_Scope, m_Input, 0.f);
  // cast the result to float so the types are compatible
  auto floatGreaterThanZero = Cast(m_Scope, greaterThanZero, tensorflow::DT_FLOAT);
  return Multiply(m_Scope, previousInputsGradient, floatGreaterThanZero);
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) { return new ReLUActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("ReLUActivation", ctor);
}

