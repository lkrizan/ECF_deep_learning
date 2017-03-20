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
	// placeholders for layer parameters
	tensorflow::ops::Placeholder *m_pWeights = nullptr;
	tensorflow::ops::Placeholder *m_pBias = nullptr;
	tensorflow::Output m_Output;
	// output shape - TODO: redefinition to Layer::TensorShape class
	std::vector<int> m_OutputShape;

	FullyConnectedLayer(const Layer &previousLayer, tensorflow::Scope &scope, std::vector<int> shape);
	~FullyConnectedLayer();

public:
	static const std::string s_LayerName;

	tensorflow::Output forward() const override;
	// TODO: redefinition to Layer::TensorShape class
	const std::vector<int>& outputShape() override;
	const std::vector<std::pair<std::string, std::vector<int>>>& getParamShapes() override;
};

int FullyConnectedLayer::s_TotalNumber = 0;
const std::string FullyConnectedLayer::s_LayerName = "FC";