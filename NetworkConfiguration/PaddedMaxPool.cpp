#include "PaddedMaxPool.h"

namespace NetworkConfiguration {

PaddedMaxPool::PaddedMaxPool(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, const std::vector<int> & windowShapeArgs, const std::vector<int> & strideShapeArgs) :
  NonParameterizedLayer(scope)
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
  m_Index = ++s_TotalNumber;
  std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";
  std::vector<tensorflow::int64> previousLayerShapeValues = previousLayerOutputShape.data();
  const unsigned int numExamples = previousLayerShapeValues[0];
  const unsigned int height = previousLayerShapeValues[1];
  const unsigned int width = previousLayerShapeValues[2];
  const unsigned int numFiltersInput = previousLayerShapeValues[3];
  const int stride = strideShapeArgs.front();
  const int poolSize = windowShapeArgs.front();
  m_OutputShape = Shape({ numExamples, height / stride, width / stride, numFiltersInput });
  using namespace tensorflow::gtl;
  m_Output = tensorflow::ops::MaxPool(scope.WithOpName(outputName), previousLayerOutput, ArraySlice<int>({ 1, poolSize, poolSize, 1 }), ArraySlice<int>({ 1, stride, stride, 1 }), tensorflow::StringPiece("SAME"));
}

}   // namespace NetworkConfiguration


// register in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new PaddedMaxPool(static_cast<LayerShapeL2Params &> (params));};
  bool dummy = LayerFactory::instance().registerClass("PaddedMaxPool", ctor);
}