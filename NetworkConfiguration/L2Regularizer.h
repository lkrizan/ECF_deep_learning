#ifndef L2Regularizer_h
#define L2Regularizer_h

#include "Layer.h"
#include "Common/Common.h"

namespace NetworkConfiguration {

class L2Regularizer
{
  tensorflow::Output m_Output;

public:
  L2Regularizer(tensorflow::Scope & scope, const std::vector<LayerP> & networkConfiguration, const std::string & outputNodeName)
  {
    std::vector<ParameterizedLayerP> parameterizedLayers;
    parameterizedLayers.reserve(std::count_if(networkConfiguration.begin(), networkConfiguration.end(), [](const LayerP layer) {return layer->hasParams();}));
    std::for_each(networkConfiguration.begin(), networkConfiguration.end(), [&parameterizedLayers](const LayerP & layer) {if (layer->hasParams())  parameterizedLayers.push_back(std::dynamic_pointer_cast<ParameterizedLayer>(layer));});
    // none of the weights shall be more than 4D
    const std::vector<int> axes = { 0, 1, 2, 3 };
    // regularization loss for first layer weights (separated so it could be done via loop)
    ParameterizedLayerP layer = parameterizedLayers.front();
    Shape shape = layer->getWeightsShape();
    auto weights = layer->getWeights();
    tensorflow::TensorShape tensorShape({ static_cast<long long>(shape.size()) });
    tensorflow::Tensor axesTensor(tensorflow::DT_INT32, tensorShape);
    Common::setTensor<int>(axesTensor, axes.begin(), axes.begin() + shape.size());
    tensorflow::Output regularizedLoss = tensorflow::ops::Sum(scope, tensorflow::ops::Square(scope, weights), axesTensor);
    // other layers (except the last one)
    for (unsigned int i = 1; i < parameterizedLayers.size() - 1; ++i)
    {
      layer = parameterizedLayers[i];
      weights = layer->getWeights();
      shape = layer->getWeightsShape();
      tensorflow::TensorShape tensorShape({ static_cast<long long>(shape.size()) });
      tensorflow::Tensor axesTensor(tensorflow::DT_INT32, tensorShape);
      Common::setTensor<int>(axesTensor, axes.begin(), axes.begin() + shape.size());
      regularizedLoss = tensorflow::ops::Add(scope, regularizedLoss, tensorflow::ops::Sum(scope, tensorflow::ops::Square(scope, weights), axesTensor));
    }
    // separate the last one so it can be named and saved as a member
    layer = parameterizedLayers.back();
    weights = layer->getWeights();
    shape = layer->getWeightsShape();
    tensorShape = tensorflow::TensorShape({ static_cast<long long>(shape.size()) });
    axesTensor = tensorflow::Tensor(tensorflow::DT_INT32, tensorShape);
    Common::setTensor<int>(axesTensor, axes.begin(), axes.begin() + shape.size());
    m_Output = tensorflow::ops::Add(scope.WithOpName(outputNodeName), regularizedLoss, tensorflow::ops::Sum(scope, tensorflow::ops::Square(scope, weights), axesTensor));
  }

  const tensorflow::Output & getOutput() const
  {
    return m_Output;
  }

};

}

#endif
