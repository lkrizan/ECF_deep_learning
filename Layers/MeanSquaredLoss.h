#ifndef MeanSquaredLoss_h
#define MeanSquaredLoss_h

#include "LossLayer.h"

namespace Layers {

class MeanSquaredLoss : public LossLayer
{
	tensorflow::Output m_Output;
	Shape m_OutputShape;

public:
	MeanSquaredLoss(const tensorflow::Input &previousLayerOutput, const tensorflow::Input & expectedOutputsPlaceholder, tensorflow::Scope &scope, Shape inputShape, Shape outputShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() const override;
};

}

#endif
