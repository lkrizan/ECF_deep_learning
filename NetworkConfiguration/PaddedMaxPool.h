#ifndef PaddedMaxPool_h
#define PaddedMaxPool_h

#include "Layer.h"

namespace NetworkConfiguration {

class PaddedMaxPool : public NonParameterizedLayer
{
  static int s_TotalNumber;
  static const std::string s_LayerName;
  // layer arguments
  int m_Stride;
  int m_PoolSize;
  tensorflow::ops::MaxPoolWithArgmax * m_pOutNode = nullptr;
  Shape m_InputShape;

public:
  // windowShape - poolSize (1D), strideShape (1D)
  PaddedMaxPool(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int>& windowShapeArgs, const std::vector<int> & strideShapeArgs);
  PaddedMaxPool(LayerShapeL2Params & params) : PaddedMaxPool(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_, params.strideShapeArgs_) {};
  ~PaddedMaxPool() { if(m_pOutNode != nullptr) delete m_pOutNode; }
  tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) override;
};


int PaddedMaxPool::s_TotalNumber = 0;
const std::string PaddedMaxPool::s_LayerName = "PaddedMaxPool";

}   // namespace NetworkConfiguration

#endif
