#include "TanhActivation.h"

namespace NetworkConfiguration {

TanhActivation::TanhActivation(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) :
  NonParameterizedLayer(scope, previousLayerOutput)
{
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Tanh(scope.WithOpName(outputName), m_Input);
}

tensorflow::Output TanhActivation::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  auto one = Const(m_Scope, 1.f);
  auto grad = Sub(m_Scope, one, Square(m_Scope, m_Output));
  return Multiply(m_Scope, grad, previousInputsGradient);
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) { return new TanhActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("TanhActivation", ctor);
}

