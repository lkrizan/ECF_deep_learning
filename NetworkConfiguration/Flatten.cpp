#include "Flatten.h"

namespace NetworkConfiguration {

Flatten::Flatten(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape) :
  NonParameterizedLayer(scope, previousLayerOutput)
{
  // check if flattening is possible (input rank 4)
  if (previousLayerOutputShape.size() != 4)
  {
    throw std::logic_error("Flattening can only be performed on rank 4 tensor inputs.");
  }

  // set name
  m_Index = ++s_TotalNumber;
  std::string outputName = s_LayerName + std::to_string(m_Index) + "_out";

  const std::vector<tensorflow::int64> & previousLayerShapeValues = previousLayerOutputShape.data();
  const unsigned int numExamples = previousLayerShapeValues[0];
  const unsigned int height = previousLayerShapeValues[1];
  const unsigned int width = previousLayerShapeValues[2];
  const unsigned int numFiltersInput = previousLayerShapeValues[3];
  const int numElements = height * width * numFiltersInput;
  m_OutputShape = Shape({ numExamples, numElements });
  m_Output = tensorflow::ops::Reshape(m_Scope.WithOpName(outputName), m_Input,
    tensorflow::Input(tensorflow::Input::Initializer({ -1, numElements })));
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new Flatten(params);};
  bool dummy = LayerFactory::instance().registerClass("Flatten", ctor);
}
