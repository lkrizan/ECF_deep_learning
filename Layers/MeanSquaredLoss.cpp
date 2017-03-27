#include "MeanSquaredLoss.h"

MeanSquaredLoss::MeanSquaredLoss(const tensorflow::Input & previousLayerOutput, tensorflow::Scope & scope, Shape inputShape, Shape outputShape)
{
	// check if inputShape to layer and outputShapes are identical
	if (inputShape != outputShape)
	{
		throw std::logic_error("Shape of expected outputs is not equal to the shape of input given to the loss function.\n");
	}

	// create graph nodes for layer
	using namespace tensorflow::ops;
	auto diff = Subtract(scope, previousLayerOutput, OUTPUTS_PLACEHOLDER_NAME);
	auto squaredDiff = Square(scope, diff);
	m_Output = Mean(scope.WithOpName(LOSS_OUTPUT_NAME), squaredDiff, 0);

	// set output shape
	m_OutputShape.push_back(1);
}

const tensorflow::Output & MeanSquaredLoss::forward() const
{
	return m_Output;
}

Shape MeanSquaredLoss::outputShape() const
{
	return m_OutputShape;
}
