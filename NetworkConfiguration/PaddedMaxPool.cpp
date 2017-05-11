#include "PaddedMaxPool.h"

namespace NetworkConfiguration {

PaddedMaxPool::PaddedMaxPool(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, const std::vector<int> & windowShapeArgs, const std::vector<int> & strideShapeArgs) :
  NonParameterizedLayer(scope, previousLayerOutput)
{
  // check if parameters are valid
  bool parameterizationFailure = false;
  std::ostringstream errorMessageStream;
  if (windowShapeArgs.size() != 1 || windowShapeArgs.front() <= 0)
  {
    parameterizationFailure = true;
    errorMessageStream << "Pool size arguments should contain one greater than zero element." << std::endl;
  }
  if (strideShapeArgs.size() != 1 || strideShapeArgs.front() <= 0)
  {
    parameterizationFailure = true;
    errorMessageStream << "Stride arguments should contain one greater than zero element." << std::endl;
  }
  if (previousLayerOutputShape.size() != 4)
  {
    parameterizationFailure = true;
    errorMessageStream << "Pooling can only be performed on tensor rank 4 inputs." << std::endl;
  }
  if (parameterizationFailure)
  {
    throw std::logic_error(errorMessageStream.str());
  }
  m_InputShape = previousLayerOutputShape;
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";
  std::vector<tensorflow::int64> previousLayerShapeValues = previousLayerOutputShape.data();
  const unsigned int numExamples = previousLayerShapeValues[0];
  const unsigned int height = previousLayerShapeValues[1];
  const unsigned int width = previousLayerShapeValues[2];
  const unsigned int numFiltersInput = previousLayerShapeValues[3];
  m_Stride = strideShapeArgs.front();
  m_PoolSize = windowShapeArgs.front();
  m_OutputShape = Shape({ numExamples, (int)std::ceil(float(height) / m_Stride), (int)std::ceil(float(width) / m_Stride), numFiltersInput });
  using namespace tensorflow::gtl;
  m_pOutNode = new tensorflow::ops::MaxPoolWithArgmax(scope.WithOpName(outputName), m_Input, ArraySlice<int>({ 1, m_PoolSize, m_PoolSize, 1 }), ArraySlice<int>({ 1, m_Stride, m_Stride, 1 }), tensorflow::StringPiece("SAME"));
  m_Output = m_pOutNode->output;
}

tensorflow::Output PaddedMaxPool::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  using namespace tensorflow::ops;
  auto indices = Reshape(m_Scope.WithOpName("indices"), m_pOutNode->argmax, {-1 , 1});
  // reshape requires shape in int32 format
  auto values = Reshape(m_Scope, previousInputsGradient, { static_cast<int>(m_OutputShape.numberOfElements()) });
  // logically, this function requires shape in int64 format
  auto result = ScatterNd(m_Scope.WithOpName("gradOut"), indices, values, {(tensorflow::int64) m_InputShape.numberOfElements()});
  return Reshape(m_Scope, result, tensorflow::ops::Shape(m_Scope, m_Input));
}

}   // namespace NetworkConfiguration


// register in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new PaddedMaxPool(static_cast<LayerShapeL2Params &> (params));};
  bool dummy = LayerFactory::instance().registerClass("PaddedMaxPool", ctor);
}