#ifndef CrossEntropyLossFunction_h
#define CrossEntropyLossFunction_h

#include "LossFunction.h"

namespace NetworkConfiguration {

class CrossEntropyLossFunction : public LossFunction
{
public:
  CrossEntropyLossFunction(tensorflow::Scope &scope, const tensorflow::Input &networkOutput, const Shape &networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape &expectedOutputShape, std::string placeholderName = "MeanSquaredLoss");
  CrossEntropyLossFunction(LossBaseParams params) :
    CrossEntropyLossFunction(params.scope_, params.networkOutput_, params.networkOutputShape_, params.expectedOutputsPlaceholder_, params.expectedOutputShape_, params.placeholderName_) {};

};

}   // namespace NetworkConfiguration

#endif
