#ifndef SoftmaxActivation_h
#define SoftmaxActivation_h

#include "Layer.h"

namespace NetworkConfiguration {

class SoftmaxActivation : public NonParameterizedLayer
{
private:
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  SoftmaxActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  SoftmaxActivation(LayerBaseParams & params) : SoftmaxActivation(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};

int SoftmaxActivation::s_TotalNumber = 0;
const std::string SoftmaxActivation::s_LayerName = "SoftmaxActivation";


}   // namespace NetworkConfiguration

#endif
