#ifndef SigmoidActivation_h
#define SigmoidActivation_h

#include "NonParameterizedLayer.h"

namespace NetworkConfiguration {

class SigmoidActivation : public NonParameterizedLayer
{
private:
  // scope for placeholder variables
  tensorflow::Scope &m_Scope;
  // placeholder for output out of the layer
  tensorflow::Output m_Output;
  // output shape
  Shape m_OutputShape;
  // used for placeholder symbolic names
  int m_Index;
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  SigmoidActivation(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape);
  const tensorflow::Output& forward() const override;
  Shape outputShape() const override;
};

int SigmoidActivation::s_TotalNumber = 0;
const std::string SigmoidActivation::s_LayerName = "SigmoidActivation";

}	// namespace NetworkConfiguration

#endif