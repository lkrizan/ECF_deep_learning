#include "SoftmaxActivation.h"

namespace NetworkConfiguration {

SoftmaxActivation::SoftmaxActivation(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) : m_Scope(scope)
{
	m_Index = ++s_TotalNumber;
	std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";
	m_OutputShape = previousLayerOutputShape;
	m_Output = tensorflow::ops::Softmax(scope.WithOpName(outputName), previousLayerOutput);
}

const tensorflow::Output & SoftmaxActivation::forward() const
{
	return m_Output;
}

Shape SoftmaxActivation::outputShape() const
{
	return m_OutputShape;
}

}	// namespace NetworkConfiguration
