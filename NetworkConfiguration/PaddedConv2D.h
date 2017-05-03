#ifndef PaddedConv2D_h
#define PaddedConv2D_h

#include "Layer.h"

namespace NetworkConfiguration {

// does not support strided convolution
class PaddedConv2D : public ParameterizedLayer
{
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

public:
  PaddedConv2D(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs);
  PaddedConv2D(LayerShapeL1Params & params) : PaddedConv2D(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_) {};
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
};

int PaddedConv2D::s_TotalNumber = 0;
const std::string PaddedConv2D::s_LayerName = "PaddedConv2D";

} // namespace NetworkConfiguration

#endif