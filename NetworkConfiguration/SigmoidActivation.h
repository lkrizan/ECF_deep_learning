#ifndef SigmoidActivation_h
#define SigmoidActivation_h

#include "NonParameterizedLayer.h"

namespace NetworkConfiguration {

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
	SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() const override;
};

}	// namespace NetworkConfiguration

#endif