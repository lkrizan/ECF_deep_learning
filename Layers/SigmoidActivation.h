#ifndef SigmoidActivation_h
#define SigmoidActivation_h

#include "NonParameterizedLayer.h"

namespace Layers {

class SigmoidActivation : public NonParameterizedLayer
{
private:
	// scope for placeholder variables
	tensorflow::Scope &m_Scope;
	// placeholder for output out of the layer
	tensorflow::Output m_Output;
	// output shape
	Shape m_OutputShape;

public:
	SigmoidActivation(const tensorflow::Input &previousLayerOutput, tensorflow::Scope &scope, Shape inputShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() const override;
};

}

#endif