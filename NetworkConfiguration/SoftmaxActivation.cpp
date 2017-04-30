#include "SoftmaxActivation.h"

namespace NetworkConfiguration {

SoftmaxActivation::SoftmaxActivation(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) : NonParameterizedLayer(scope)
{
  m_Index = ++s_TotalNumber;
  std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";
  m_OutputShape = previousLayerOutputShape;
  m_Output = tensorflow::ops::Softmax(scope.WithOpName(outputName), previousLayerOutput);
}

}	// namespace NetworkConfiguration

  // register class in LayerFactory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new SoftmaxActivation(params);};
  bool dummy = LayerFactory::instance().registerClass("SoftmaxActivation", ctor);
}
