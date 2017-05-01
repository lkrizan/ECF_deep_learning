#include "Conv2D.h"

namespace NetworkConfiguration {

  // paramShape: kernelSize, numFilters
  Conv2D::Conv2D(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, const Shape & paramShape, const Shape & strideShape) :
    ParameterizedLayer(scope)
  {
    using namespace tensorflow::ops;
    // check if parameters are valid
    bool parameterizationFailure = false;
    std::ostringstream errorMessageStream;
    if (!paramShape.validForParameterizedUse() || paramShape.size() != 2)
    {
      parameterizationFailure = true;
      errorMessageStream << "Shape [" << paramShape << "] cannot be used as a kernel for convolution." << std::endl;
    }
    if (previousLayerOutputShape.size() != 4)
    {
      parameterizationFailure = true;
      errorMessageStream << "Input to a convolution layer must be a rank 4 tensor" << std::endl;
    }
    // check if compatible for matrix multiplication ( forward pass is x dot wT
    else if (!strideShape.validForParameterizedUse() || strideShape.size() != 1)
    {
      parameterizationFailure = true;
      errorMessageStream << "Shapes [" << strideShape << "] is not a valid shape for stride. Only one element allowed." << std::endl;
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
    // get data from shapes to create variables
    const int stride = strideShape.front();
    const unsigned int kernelSize = paramShape.front();
    const unsigned int numFilters = paramShape.back();
    const std::vector<int64> & previousLayerShapeValues = previousLayerOutputShape.data();
    const unsigned int numExamples = previousLayerShapeValues[0];
    const unsigned int height = previousLayerShapeValues[1];
    const unsigned int width = previousLayerShapeValues[2];
    const unsigned int numFiltersInput = previousLayerShapeValues[3];
    m_OutputShape = Shape({ numExamples, height - kernelSize + 1, width - kernelSize + 1, numFilters });
    m_WeightsShape = Shape({ kernelSize, kernelSize, numFiltersInput, numFilters });
    m_BiasShape = Shape({ numFilters });
    tensorflow::gtl::ArraySlice<int> finalStrideShape = { 1, stride, stride, 1 };
    // create placeholders and graph nodes for layer 
    auto weights = Placeholder(m_Scope.WithOpName(m_WeightsName), tensorflow::DataType::DT_FLOAT);
    auto bias = Placeholder(m_Scope.WithOpName(m_BiasName), tensorflow::DataType::DT_FLOAT);
    auto tempResult = tensorflow::ops::Conv2D(m_Scope, previousLayerOutput, weights, finalStrideShape, tensorflow::StringPiece("VALID"));
    m_Output = Add(m_Scope.WithOpName(name + "_out"), tempResult, bias);
  }

  std::vector<std::pair<std::string, Shape>> Conv2D::getParamShapes() const
  {
    return std::vector<std::pair<std::string, Shape>>({ { m_WeightsName, m_WeightsShape },{ m_BiasName, m_BiasShape } });
  }

}   // namespace NetworkConfiguration

// register class in factory
namespace {
  using namespace NetworkConfiguration;
  LayerCreator ctor = [](LayerBaseParams & params) {return new Conv2D(static_cast<LayerShapeL2Params&>(params));};
  bool dummy = LayerFactory::instance().registerClass("Conv2D", ctor);
}
