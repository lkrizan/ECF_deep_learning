#include "Layer.h"

class SigmoidLayer : NonParameterizedLayer
{
private:
	// scope for placeholder variables
	tensorflow::Scope &m_Scope;
	// placeholder for output out of the layer
	tensorflow::Output m_Output;
	// output shape
	Shape m_OutputShape;

public:
	SigmoidLayer(const tensorflow::Input &previousLayerOutput, tensorflow::Scope &scope, Shape inputShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() const override;
};