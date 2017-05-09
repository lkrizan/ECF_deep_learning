#ifndef FullyConnectedLayer_h
#define FullyConnectedLayer_h

#include "Layer.h"

namespace NetworkConfiguration {

class FullyConnectedLayer : public ParameterizedLayer
{
private:
  // counter of class instances
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // parameters name
  std::string m_WeightsName;
  std::string m_BiasName;

public:
  // paramShape - number of neurons
  FullyConnectedLayer(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs);
  FullyConnectedLayer(LayerShapeL1Params & params);
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
  tensorflow::Output backwardWeights(const tensorflow::Input & previousInputsGradient) override;
  tensorflow::Output backwardBias(const tensorflow::Input & previousInputsGradient) override;
};

int FullyConnectedLayer::s_TotalNumber = 0;
const std::string FullyConnectedLayer::s_LayerName = "FC";

}	// namespace NetworkConfiguration

#endif