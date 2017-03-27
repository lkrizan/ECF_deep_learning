#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer(const tensorflow::Input & previousLayerOutput, tensorflow::Scope & scope, Shape inputShape) : m_Scope(scope)
{
	m_OutputShape = inputShape;
	m_Output = tensorflow::ops::Sigmoid(scope, previousLayerOutput);
}

const tensorflow::Output & SigmoidLayer::forward() const
{
	return m_Output;
}

Shape SigmoidLayer::outputShape() const
{
	return m_OutputShape;
}


