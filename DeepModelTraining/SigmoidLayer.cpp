#include "SigmoidLayer.h"

SigmoidActivation::SigmoidActivation(const tensorflow::Input & previousLayerOutput, tensorflow::Scope & scope, Shape inputShape) : m_Scope(scope)
{
	m_OutputShape = inputShape;
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


