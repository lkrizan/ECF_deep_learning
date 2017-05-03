#include "Conv2D.h"

namespace NetworkConfiguration {

  // paramShape: kernelSize, numFilters
  Conv2D::Conv2D(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput, const Shape & previousLayerOutputShape, const std::vector<int> & paramShapeArgs) :
    ParameterizedLayer(scope)
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
    const unsigned int kernelSize = paramShapeArgs.front();
    const unsigned int numFilters = paramShapeArgs.back();
    const std::vector<int64> & previousLayerShapeValues = previousLayerOutputShape.data();
    const unsigned int numExamples = previousLayerShapeValues[0];
    const unsigned int height = previousLayerShapeValues[1];
    const unsigned int width = previousLayerShapeValues[2];
    const unsigned int numFiltersInput = previousLayerShapeValues[3];
    m_OutputShape = Shape({ numExamples, height - kernelSize + 1, width - kernelSize + 1, numFilters });
    m_WeightsShape = Shape({ kernelSize, kernelSize, numFiltersInput, numFilters });
    m_BiasShape = Shape({ numFilters });
    // create placeholders and graph nodes for layer 
    auto weights = Placeholder(m_Scope.WithOpName(m_WeightsName), tensorflow::DataType::DT_FLOAT);
    auto bias = Placeholder(m_Scope.WithOpName(m_BiasName), tensorflow::DataType::DT_FLOAT);
    // for some reason, build with optimization (max speed) throws exception unless array slice which describes stride is passed directly through function (?)
    auto tempResult = tensorflow::ops::Conv2D(m_Scope, previousLayerOutput, weights, tensorflow::gtl::ArraySlice<int>({1, 1, 1, 1}), tensorflow::StringPiece("VALID"));
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
  LayerCreator ctor = [](LayerBaseParams & params) {return new Conv2D(static_cast<LayerShapeL1Params&>(params));};
  bool dummy = LayerFactory::instance().registerClass("Conv2D", ctor);
}
