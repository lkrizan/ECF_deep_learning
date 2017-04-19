#ifndef FullyConnectedLayer_h
#define FullyConnectedLayer_h

#include "ParameterizedLayer.h"

namespace NetworkConfiguration {

class FullyConnectedLayer : public ParameterizedLayer
{
private:
  // scope for placeholder variables
  tensorflow::Scope &m_Scope;
  // index of this layer - used for unique variable names
  int m_Index;
  // counter of class instances
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // parameters name
  std::string m_WeightsName;
  std::string m_BiasName;
  // parameters shape
  Shape m_WeightsShape;
  Shape m_BiasShape;
  Shape m_OutputShape;
  // placeholder for output out of the layer
  tensorflow::Output m_Output;

public:
  FullyConnectedLayer(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const Shape &paramShape);
  const tensorflow::Output& forward() const override;
  Shape outputShape() const override;
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
};

int FullyConnectedLayer::s_TotalNumber = 0;
const std::string FullyConnectedLayer::s_LayerName = "FC";

}	// namespace NetworkConfiguration

#endif