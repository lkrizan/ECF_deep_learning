#ifndef SigmoidActivation_h
#define SigmoidActivation_h

#include "Layer.h"

namespace NetworkConfiguration {

class SigmoidActivation : public NonParameterizedLayer
{
private:
  // used for placeholder symbolic names
  int m_Index;
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  SigmoidActivation(LayerBaseParams & params) : SigmoidActivation(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};

  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};

int SigmoidActivation::s_TotalNumber = 0;
const std::string SigmoidActivation::s_LayerName = "SigmoidActivation";

}	// namespace NetworkConfiguration

#endif