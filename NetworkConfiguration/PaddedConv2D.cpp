#include "PaddedConv2D.h"

namespace NetworkConfiguration {

// paramShape: kernelSize, numFilters
PaddedConv2D::PaddedConv2D(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, 
  const std::vector<int> & paramShapeArgs, const std::vector<int> & strideShapeArgs) :
  ParameterizedLayer(scope, previousLayerOutput)
{
  using namespace tensorflow::ops;
  // check if parameters are valid
  bool parameterizationFailure = false;
  std::ostringstream errorMessageStream;
  if (paramShapeArgs.size() != 2 || paramShapeArgs[0] <= 0 || paramShapeArgs[1] <= 0)
  {
    parameterizationFailure = true;
    errorMessageStream << "Convolution parameters should have 2 greater than zero arguments." << std::endl;
  }
  if (previousLayerOutputShape.size() != 4)
  {
    parameterizationFailure = true;
    errorMessageStream << "Input to a convolution layer must be a rank 4 tensor" << std::endl;
  }
  if (strideShapeArgs.size() != 1 || strideShapeArgs.front() <= 0)
  {
    parameterizationFailure = true;
    errorMessageStream << "Stride parameter should be 1 greater than zero argument." << std::endl;
  }
  if (parameterizationFailure)
  {
    throw std::logic_error(errorMessageStream.str());
  }
  // set names for variables
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  m_WeightsName = m_LayerName + "_w";
  m_BiasName = m_LayerName + "_b";
  // get data from shapes to create variables
  const unsigned int kernelSize = paramShapeArgs.front();
  const unsigned int numFilters = paramShapeArgs.back();
  const std::vector<int64> & previousLayerShapeValues = previousLayerOutputShape.data();
  const unsigned int numExamples = previousLayerShapeValues[0];
  const unsigned int height = previousLayerShapeValues[1];
  const unsigned int width = previousLayerShapeValues[2];
  const unsigned int numFiltersInput = previousLayerShapeValues[3];
  m_Stride = strideShapeArgs.front();
  m_OutputShape = Shape({ numExamples, (int)std::ceil(float(height) / m_Stride), (int)std::ceil(float(width) / m_Stride), numFilters });
  m_WeightsShape = Shape({ kernelSize, kernelSize, numFiltersInput, numFilters });
  m_BiasShape = Shape({ numFilters });
  // create placeholders and graph nodes for layer 
  m_Weights = Placeholder(m_Scope.WithOpName(m_WeightsName), tensorflow::DataType::DT_FLOAT);
  m_Bias = Placeholder(m_Scope.WithOpName(m_BiasName), tensorflow::DataType::DT_FLOAT);
  // for some reason, build with optimization (max speed) throws exception unless array slice which describes stride is passed directly through function (?)
  auto tempResult = Conv2D(m_Scope, m_Input, m_Weights, tensorflow::gtl::ArraySlice<int>({ 1, m_Stride, m_Stride, 1 }), tensorflow::StringPiece("SAME"));
  m_Output = BiasAdd(m_Scope.WithOpName(m_LayerName + "_out"), tempResult, m_Bias);
}

std::vector<std::pair<std::string, Shape>> PaddedConv2D::getParamShapes() const
{
  return std::vector<std::pair<std::string, Shape>>({ { m_WeightsName, m_WeightsShape },{ m_BiasName, m_BiasShape } });
}

tensorflow::Output PaddedConv2D::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  using tensorflow::ops::Shape;
  return Conv2DBackpropInput(m_Scope, Shape(m_Scope, m_Input), m_Weights, previousInputsGradient, tensorflow::gtl::ArraySlice<int>({ 1, m_Stride, m_Stride, 1 }), tensorflow::StringPiece("SAME"));
}

tensorflow::Output PaddedConv2D::backwardWeights(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  using tensorflow::ops::Shape;
  return Conv2DBackpropFilter(m_Scope, m_Input, Shape(m_Scope, m_Weights), previousInputsGradient, tensorflow::gtl::ArraySlice<int>({ 1, m_Stride, m_Stride, 1 }), tensorflow::StringPiece("SAME"));
}

tensorflow::Output PaddedConv2D::backwardBias(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  return BiasAddGrad(m_Scope, previousInputsGradient);
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new PaddedConv2D(static_cast<LayerShapeL2Params&>(params));};
  bool dummy = LayerFactory::instance().registerClass("PaddedConv2D", ctor);
}

