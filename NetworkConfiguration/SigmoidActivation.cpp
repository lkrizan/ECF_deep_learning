#include "SigmoidActivation.h"

namespace NetworkConfiguration {

SigmoidActivation::SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape) : 
  NonParameterizedLayer(scope, previousLayerOutput)
{
  m_Index = ++s_TotalNumber;
  std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Sigmoid(scope.WithOpName(outputName), m_Input);
}

tensorflow::Output SigmoidActivation::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  auto subConstant = Const(m_Scope, 1.f);
  auto sigmoidGradInputs = Sigmoid(m_Scope, previousInputsGradient);
  auto temp = Subtract(m_Scope, subConstant, sigmoidGradInputs);
  return Multiply(m_Scope, sigmoidGradInputs, temp);
}

}	// namespace NetworkConfiguration

// register class in LayerFactory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new SigmoidActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("SigmoidActivation", ctor);
}


