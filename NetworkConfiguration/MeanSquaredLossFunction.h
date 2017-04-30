#ifndef MeanSquaredLossFunction_h
#define MeanSquaredLossFunction_h

#include "LossFunction.h"

namespace NetworkConfiguration {

class MeanSquaredLossFunction : public LossFunction
{
public:
  MeanSquaredLossFunction(tensorflow::Scope &scope, const tensorflow::Input &networkOutput, const Shape &networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape &expectedOutputShape, std::string placeholderName="MeanSquaredLoss");
  MeanSquaredLossFunction(LossBaseParams params) :
    MeanSquaredLossFunction(params.scope_, params.networkOutput_, params.networkOutputShape_, params.expectedOutputsPlaceholder_, params.expectedOutputShape_, params.placeholderName_) {};
};

}	// namespace NetworkConfiguration

#endif
