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

  m_InputShape = previousLayerOutputShape;
  // set name
  m_LayerName = s_LayerName + std::to_string(++s_TotalNumber);
  std::string outputName = m_LayerName + "_out";

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

tensorflow::Output Flatten::backwardInputs(const tensorflow::Input & previousInputsGradient)
{
  // reshape back to original shape
  using namespace tensorflow::ops;
  std::vector<int> inputShapeValues;
  inputShapeValues.reserve(4);
  std::copy(m_InputShape.begin(), m_InputShape.end(), std::back_inserter(inputShapeValues));
  return Reshape(m_Scope, previousInputsGradient, { inputShapeValues[0], inputShapeValues[1], inputShapeValues[2], inputShapeValues[3] });
}

}   // namespace NetworkConfiguration


// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new Flatten(params);};
  bool dummy = LayerFactory::instance().registerClass("Flatten", ctor);
}
