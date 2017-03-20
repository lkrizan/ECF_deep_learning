#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(const Layer &previousLayer, tensorflow::Scope &scope, std::vector<int> shape) : m_Scope(scope)
{
	using namespace tensorflow::ops;

	// TODO: check given shape - maybe create exception class?
	// TODO: check if shape of this layer is compatible with previous for operations

	// set names for variables
	m_Index = ++s_TotalNumber;
	std::string name = s_LayerName + std::to_string(m_Index);

	// create placeholders and graph nodes for layer 
	m_pWeights = new Placeholder(m_Scope.WithOpName(name + "_w"), tensorflow::DataType::DT_FLOAT);
	m_pBias = new Placeholder(m_Scope.WithOpName(name + "_b"), tensorflow::DataType::DT_FLOAT);
	auto tempResult = MatMul(m_Scope, previousLayer.forward(), *m_pWeights, MatMul::TransposeB(true));
	m_Output = Add(m_Scope, tempResult, *m_pBias);

	// TODO: with Layer::TensorShape class
	m_OutputShape.push_back(shape[0]);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
	if (m_pWeights != nullptr)
		delete m_pWeights;
	if (m_pBias != nullptr)
		delete m_pBias;
}

tensorflow::Output FullyConnectedLayer::forward() const
{
	return m_Output;
}

const std::vector<int>& FullyConnectedLayer::outputShape()
{
	// TODO: insert return statement here
}

const std::vector<std::pair<std::string, std::vector<int>>>& FullyConnectedLayer::getParamShapes()
{
	// TODO: insert return statement here
}
