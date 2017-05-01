#ifndef PaddedMaxPool_h
#define PaddedMaxPool_h

#include "Layer.h"

namespace NetworkConfiguration {

class PaddedMaxPool : public NonParameterizedLayer
{
  // used for placeholder symbolic names
  int m_Index;
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  // windowShape - poolSize (1D), strideShape (1D)
  PaddedMaxPool(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const Shape& windowShape, const Shape & strideShape);
  PaddedMaxPool(LayerShapeL2Params & params) : PaddedMaxPool(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShape_, params.strideShape_) {};
};


int PaddedMaxPool::s_TotalNumber = 0;
const std::string PaddedMaxPool::s_LayerName = "PaddedMaxPool";

}   // namespace NetworkConfiguration

#endif
