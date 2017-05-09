#ifndef Conv2D_h
#define Conv2D_h

#include "Layer.h"

namespace NetworkConfiguration {

class Conv2D : public ParameterizedLayer
{
  // counter of class instances
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // parameters name
  std::string m_WeightsName;
  std::string m_BiasName;
  // layer arguments
  int m_Stride;

public:
  Conv2D(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs, const std::vector<int> & strideShapeArgs);
  Conv2D(LayerShapeL2Params & params) : Conv2D(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_, params.strideShapeArgs_) {};
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
};

int Conv2D::s_TotalNumber = 0;
const std::string Conv2D::s_LayerName = "Conv2D";

} // namespace NetworkConfiguration

#endif
