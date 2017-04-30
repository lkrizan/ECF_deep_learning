#ifndef Layers_all_h
#define Layers_all_h

#include "Layer.h"
#include "LossFunction.h"
#include "FullyConnectedLayer.h"
#include "MeanSquaredLossFunction.h"
#include "SigmoidActivation.h"


namespace NetworkConfiguration {

class LayerFactory
{
public:
  static LayerP createLayer(std::string layerTypeName, tensorflow::Scope& scope, const tensorflow::Input &previousLayerOutput, const Shape &previousLayerOutputShape, Shape paramShape = Shape())
  {
    if (layerTypeName == "FullyConnectedLayer")
    {
      return LayerP(new FullyConnectedLayer(scope, previousLayerOutput, previousLayerOutputShape, paramShape));
    }
    else if (layerTypeName == "SigmoidActivation")
    {
      return LayerP(new SigmoidActivation(scope, previousLayerOutput, previousLayerOutputShape));
    }
    else
    {
      throw std::logic_error(layerTypeName + " is not a registered layer factory.\n");
    }
  }
};


class LossFunctionFactory {
public:
  static LossFunctionP createLossFunction(std::string lossFunctionTypeName, tensorflow::Scope& scope, const tensorflow::Input &networkOutput, const Shape &networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName)
  {
    if (lossFunctionTypeName == "MeanSquaredLossFunction")
    {
      return LossFunctionP(new MeanSquaredLossFunction(scope, networkOutput, networkOutputShape, expectedOutputsPlaceholder, expectedOutputShape, placeholderName));
    }
    else
    {
      throw std::logic_error(lossFunctionTypeName + " is not a registered loss function factory.\n");
    }
  }
};

}

#endif
