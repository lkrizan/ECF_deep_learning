#ifndef ReLUActivation_h
#define ReLUActivation_h

#include "Layer.h"

namespace NetworkConfiguration {

class ReLUActivation : public NonParameterizedLayer
{
private:
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  ReLUActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  ReLUActivation(LayerBaseParams & params) : ReLUActivation(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};

int ReLUActivation::s_TotalNumber = 0;
const std::string ReLUActivation::s_LayerName = "ReLUActivation";


}   // namespace NetworkConfiguration

#endif
