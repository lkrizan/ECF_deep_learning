#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(const tensorflow::Input &previousLayerOutput, tensorflow::Scope &scope, Shape inputShape, Shape paramShape) : m_Scope(scope)
{
	using namespace tensorflow::ops;

	// TODO: check given shape - maybe create exception class?
	// TODO: check if shape of this layer is compatible with previous for operations

	// set names for variables
	m_Index = ++s_TotalNumber;
	std::string name = s_LayerName + std::to_string(m_Index);
	m_WeightsName = name + "_w";
	m_BiasName = name + "_b";

	// create placeholders and graph nodes for layer 
	auto weights = Placeholder(m_Scope.WithOpName(m_WeightsName), tensorflow::DataType::DT_FLOAT);
	auto bias = Placeholder(m_Scope.WithOpName(m_BiasName), tensorflow::DataType::DT_FLOAT);
	auto tempResult = MatMul(m_Scope, previousLayerOutput, weights, MatMul::TransposeB(true));
	m_Output = Add(m_Scope, tempResult, bias);

	// set shapes
	m_WeightsShape = paramShape;
	m_WeightsShape.transpose();
	m_BiasShape.push_back(m_WeightsShape.front());
	m_OutputShape = inputShape;
	m_OutputShape.push_back(m_WeightsShape.back());
}

const tensorflow::Output& FullyConnectedLayer::forward() const
{
	return m_Output;
}

Shape FullyConnectedLayer::outputShape()
{
	return m_OutputShape;
}

std::vector<std::pair<std::string, Shape>> FullyConnectedLayer::getParamShapes() const
{
	return std::vector<std::pair<std::string, Shape>>({ {m_WeightsName, m_WeightsShape}, {m_BiasName, m_BiasShape} });
}
