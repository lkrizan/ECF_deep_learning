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
	static const std::string s_LayerName;
	SigmoidLayer(const tensorflow::Input &previousLayerOutput, tensorflow::Scope &scope, Shape inputShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() override;
};
const std::string SigmoidLayer::s_LayerName = "Sigmoid";