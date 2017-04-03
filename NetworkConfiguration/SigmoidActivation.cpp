#include "SigmoidActivation.h"

namespace NetworkConfiguration {

SigmoidActivation::SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape) : m_Scope(scope)
{
	m_OutputShape = previousLayerOutputShape;
	m_Output = tensorflow::ops::Sigmoid(scope, previousLayerOutput);
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


