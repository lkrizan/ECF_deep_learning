#include "PaddedAvgPool.h"

namespace NetworkConfiguration {

PaddedAvgPool::PaddedAvgPool(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, const Shape & windowShape, const Shape & strideShape) :
  NonParameterizedLayer(scope)
{
  // check if parameters are valid
  bool parameterizationFailure = false;
  std::ostringstream errorMessageStream;
  if (!windowShape.validForParameterizedUse() || windowShape.size() != 4)
  {
    parameterizationFailure = true;
    errorMessageStream << "Shape [" << windowShape << "] is not a valid shape for pooling window." << std::endl;
  }
  if (!strideShape.validForParameterizedUse() || strideShape.size() != 4 || strideShape.front() != 1 || strideShape.back() != 1)
  {
    parameterizationFailure = true;
    errorMessageStream << "Shape [" << windowShape << "] is not a valid stride parameter." << std::endl;
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
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::AvgPool(scope.WithOpName(outputName), previousLayerOutput, windowShape.asArraySlice<int>(), strideShape.asArraySlice<int>(), tensorflow::StringPiece("SAME"));
}

}   // namespace NetworkConfiguration


// register in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new PaddedAvgPool(static_cast<LayerShapeL2Params &> (params));};
  bool dummy = LayerFactory::instance().registerClass("PaddedAvgPool", ctor);
}