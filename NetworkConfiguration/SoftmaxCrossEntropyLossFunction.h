#ifndef SoftmaxCrossEntropyLossFunction_h
#define SoftmaxCrossEntropyLossFunction_h

#include "LossFunction.h"

namespace NetworkConfiguration {

  class SoftmaxCrossEntropyLossFunction : public LossFunction
  {
  public:
    SoftmaxCrossEntropyLossFunction(tensorflow::Scope &scope, const tensorflow::Input &networkOutput, const Shape &networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape &expectedOutputShape, std::string placeholderName = "MeanSquaredLoss");
    SoftmaxCrossEntropyLossFunction(LossBaseParams params) :
      SoftmaxCrossEntropyLossFunction(params.scope_, params.networkOutput_, params.networkOutputShape_, params.expectedOutputsPlaceholder_, params.expectedOutputShape_, params.placeholderName_) {};

  };

}   // namespace NetworkConfiguration

#endif
