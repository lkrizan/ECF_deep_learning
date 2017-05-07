#ifndef PaddedConv2D_h
#define PaddedConv2D_h

#include "Layer.h"

namespace NetworkConfiguration {

class PaddedConv2D : public ParameterizedLayer
{
  // counter of class instances
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // parameters name
  std::string m_WeightsName;
  std::string m_BiasName;
  // parameters shape
  Shape m_WeightsShape;
  Shape m_BiasShape;
  // layer arguments
  int m_Stride;

public:
  PaddedConv2D(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs, const std::vector<int> & strideShapeArgs);
  PaddedConv2D(LayerShapeL2Params & params) : PaddedConv2D(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_, params.strideShapeArgs_) {};
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
};

int PaddedConv2D::s_TotalNumber = 0;
const std::string PaddedConv2D::s_LayerName = "PaddedConv2D";

} // namespace NetworkConfiguration

#endif