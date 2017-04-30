#ifndef SoftmaxActivation_h
#define SoftmaxActivation_h

#include "Layer.h"

namespace NetworkConfiguration {

class SoftmaxActivation : public NonParameterizedLayer
{
private:
  // used for placeholder symbolic names
  int m_Index;
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  SoftmaxActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  SoftmaxActivation(LayerBaseParams & params) : SoftmaxActivation(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};
};

int SoftmaxActivation::s_TotalNumber = 0;
const std::string SoftmaxActivation::s_LayerName = "SoftmaxActivation";

}	// namespace NetworkConfiguration

#endif
