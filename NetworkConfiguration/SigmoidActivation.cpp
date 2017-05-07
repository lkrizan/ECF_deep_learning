#include "SigmoidActivation.h"

namespace NetworkConfiguration {

SigmoidActivation::SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape) : 
  NonParameterizedLayer(scope, previousLayerOutput)
{
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Sigmoid(scope.WithOpName(outputName), m_Input);
}

tensorflow::Output SigmoidActivation::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  auto subConstant = Const(m_Scope, 1.f);
  auto tempMul = Subtract(m_Scope, subConstant, m_Output);
  auto temp =  Multiply(m_Scope, m_Output, tempMul);
  return Multiply(m_Scope, temp, previousInputsGradient);
}

}	// namespace NetworkConfiguration

// register class in LayerFactory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new SigmoidActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("SigmoidActivation", ctor);
}


