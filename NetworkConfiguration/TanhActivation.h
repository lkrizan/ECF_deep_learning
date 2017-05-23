#ifndef Tanhctivation_h
#define TanhActivation_h

#include "Layer.h"

namespace NetworkConfiguration {

class TanhActivation : public NonParameterizedLayer
{
private:
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  TanhActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  TanhActivation(LayerBaseParams & params) : TanhActivation(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_) {};
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};

int TanhActivation::s_TotalNumber = 0;
const std::string TanhActivation::s_LayerName = "TanhActivation";


}   // namespace NetworkConfiguration

#endif#pragma once
