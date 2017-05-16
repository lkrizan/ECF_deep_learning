#ifndef Flatten_h
#define Flatten_h

#include "Layer.h"

namespace NetworkConfiguration {

// flattens every learning example to a vector
class Flatten : public NonParameterizedLayer
{
private:
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // needed for gradient calculation
  Shape m_InputShape;

public:
  Flatten(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  Flatten(LayerBaseParams & params) : Flatten(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};

int Flatten::s_TotalNumber = 0;
const std::string Flatten::s_LayerName = "Flatten";

}	// namespace NetworkConfiguration

#endif