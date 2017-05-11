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
  // layer arguments
  int m_Stride;
  // input shape, needed for gradient calculation
  Shape m_InputShape;

public:
  PaddedConv2D(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int> &paramShapeArgs, const std::vector<int> & strideShapeArgs);
  PaddedConv2D(LayerShapeL2Params & params) : PaddedConv2D(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_, params.strideShapeArgs_) {};
  std::vector<std::pair<std::string, Shape>> getParamShapes() const override;
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
  tensorflow::Output backwardWeights(const tensorflow::Input & previousInputsGradient) override;
  tensorflow::Output backwardBias(const tensorflow::Input & previousInputsGradient) override;
};

int PaddedConv2D::s_TotalNumber = 0;
const std::string PaddedConv2D::s_LayerName = "PaddedConv2D";

} // namespace NetworkConfiguration

#endif