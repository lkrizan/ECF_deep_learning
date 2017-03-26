#include "Layer.h"

class FullyConnectedLayer : public ParameterizedLayer
{
private:
	// scope for placeholder variables
	tensorflow::Scope &m_Scope;
	// index of this layer - used for unique variable names
	int m_Index;
	// counter of class instances
	static int s_TotalNumber;
	// parameters shape
	Shape m_WeightsShape;
	Shape m_BiasShape;
	Shape m_OutputShape;
	// placeholder for output out of the layer
	tensorflow::Output m_Output;

public:
	static const std::string s_LayerName;
	FullyConnectedLayer(const tensorflow::Input &previousLayerOutput, tensorflow::Scope &scope, Shape inputShape, Shape paramShape);
	const tensorflow::Output& forward() const override;
	Shape outputShape() override;
	std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
};

int FullyConnectedLayer::s_TotalNumber = 0;
const std::string FullyConnectedLayer::s_LayerName = "FC";