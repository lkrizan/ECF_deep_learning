#include "FullyConnectedLayer.h"

namespace NetworkConfiguration {

FullyConnectedLayer::FullyConnectedLayer(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs) : 
  ParameterizedLayer(scope, previousLayerOutput)
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
  
  // set shapes
  unsigned int numDimension = previousLayerOutputShape.back();
  unsigned int numNeurons = paramShapeArgs.front();
  m_WeightsShape = Shape({ numNeurons, numDimension });
  m_BiasShape = Shape({ numNeurons });
  m_OutputShape = Shape({ previousLayerOutputShape.front(), m_WeightsShape.front() });

  // create placeholders and graph nodes for layer 
  m_Weights = Variable(m_Scope.WithOpName(m_WeightsName), m_WeightsShape.asTensorShape(), tensorflow::DataType::DT_FLOAT);
  m_Bias = Variable(m_Scope.WithOpName(m_BiasName), m_BiasShape.asTensorShape(), tensorflow::DataType::DT_FLOAT);
  auto tempResult = MatMul(m_Scope, m_Input, m_Weights, MatMul::TransposeB(true));
  m_Output = BiasAdd(m_Scope.WithOpName(name + "_out"), tempResult, m_Bias);

}

FullyConnectedLayer::FullyConnectedLayer(LayerShapeL1Params & params) :
  FullyConnectedLayer(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_) {};

std::vector<std::pair<std::string, Shape>> FullyConnectedLayer::getParamShapes() const
{
  return std::vector<std::pair<std::string, Shape>>({ {m_WeightsName, m_WeightsShape}, {m_BiasName, m_BiasShape} });
}

const tensorflow::Output & FullyConnectedLayer::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  return MatMul(m_Scope, previousInputsGradient, m_Weights);
}

const tensorflow::Output & FullyConnectedLayer::backwardWeights(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  return MatMul(m_Scope, previousInputsGradient, m_Input, MatMul::TransposeA(true));
}

const tensorflow::Output & FullyConnectedLayer::backwardBias(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  return Sum(m_Scope, previousInputsGradient, 0);
}

} // namespace NetworkConfiguration


// register class in LayerFactory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new FullyConnectedLayer(static_cast<LayerShapeL1Params &>(params));};
  bool dummy = LayerFactory::instance().registerClass("FullyConnectedLayer", ctor);
}