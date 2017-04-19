#include "SigmoidActivation.h"

namespace NetworkConfiguration {

SigmoidActivation::SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape) : m_Scope(scope)
{
  m_Index = ++s_TotalNumber;
  std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Sigmoid(scope.WithOpName(outputName), previousLayerOutput);
}

const tensorflow::Output & SigmoidActivation::forward() const
{
  return m_Output;
}

Shape SigmoidActivation::outputShape() const
{
  return m_OutputShape;
}

}	// namespace NetworkConfiguration


