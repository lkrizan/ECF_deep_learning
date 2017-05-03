#include "FullyConnectedLayer.h"

namespace NetworkConfiguration {

FullyConnectedLayer::FullyConnectedLayer(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs) : ParameterizedLayer(scope)
{
  using namespace tensorflow::ops;
  
  // check if parameters are valid
  bool parameterizationFailure = false;
  std::ostringstream errorMessageStream;
  if (paramShapeArgs.size() != 1 || paramShapeArgs.front() <= 0)
  {
    parameterizationFailure = true;
    errorMessageStream << "Fully connected layer shape argument must be single greater than zero element." << std::endl;
  }
  if (previousLayerOutputShape.size() != 2)
  {
    parameterizationFailure = true;
    errorMessageStream << "Input to a fully connected layer must be a rank 2 tensor." << std::endl;
  }
  if (parameterizationFailure) 
  {
    throw std::logic_error(errorMessageStream.str());
  }
  // set names for variables
  m_Index = ++s_TotalNumber;
  std::string name = s_LayerName + std::to_string(m_Index);
  m_WeightsName = name + "_w";
  m_BiasName = name + "_b";
  
  // create placeholders and graph nodes for layer 
  auto weights = Placeholder(m_Scope.WithOpName(m_WeightsName), tensorflow::DataType::DT_FLOAT);
  auto bias = Placeholder(m_Scope.WithOpName(m_BiasName), tensorflow::DataType::DT_FLOAT);
  auto tempResult = MatMul(m_Scope, previousLayerOutput, weights, MatMul::TransposeB(true));
  m_Output = Add(m_Scope.WithOpName(name + "_out"), tempResult, bias);

  // set shapes
  unsigned int numDimension = previousLayerOutputShape.back();
  unsigned int numNeurons = paramShapeArgs.front();
  m_WeightsShape = Shape({ numNeurons, numDimension });
  m_BiasShape = Shape({ numNeurons });
  m_OutputShape = Shape({ previousLayerOutputShape.front(), m_WeightsShape.front() });
}

FullyConnectedLayer::FullyConnectedLayer(LayerShapeL1Params & params) :
  FullyConnectedLayer(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_) {};

std::vector<std::pair<std::string, Shape>> FullyConnectedLayer::getParamShapes() const
{
  return std::vector<std::pair<std::string, Shape>>({ {m_WeightsName, m_WeightsShape}, {m_BiasName, m_BiasShape} });
}

} // namespace NetworkConfiguration


// register class in LayerFactory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new FullyConnectedLayer(static_cast<LayerShapeL1Params &>(params));};
  bool dummy = LayerFactory::instance().registerClass("FullyConnectedLayer", ctor);
}