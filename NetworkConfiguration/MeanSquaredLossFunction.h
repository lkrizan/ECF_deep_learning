#ifndef MeanSquaredLossFunction_h
#define MeanSquaredLossFunction_h

#include "LossFunction.h"

namespace NetworkConfiguration {

class MeanSquaredLossFunction : public LossFunction
{
  tensorflow::Output m_Loss;
  Shape m_OutputShape;

public:
  MeanSquaredLossFunction(tensorflow::Scope &scope, const tensorflow::Input &networkOutput, const Shape &networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape &expectedOutputShape, std::string placeholderName="MeanSquaredLoss");
  const tensorflow::Output& getLossOutput() const override;
  Shape outputShape() const override;
};

}	// namespace NetworkConfiguration

#endif
