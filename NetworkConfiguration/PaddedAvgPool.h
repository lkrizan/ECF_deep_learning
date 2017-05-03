#ifndef PaddedAvgPool_h
#define PaddedAvgPool_h

#include "Layer.h"

namespace NetworkConfiguration {

class PaddedAvgPool : public NonParameterizedLayer
{
  // used for placeholder symbolic names
  int m_Index;
  static int s_TotalNumber;
  static const std::string s_LayerName;

public:
  // windowShape - poolSize (1D), strideShape (1D)
  PaddedAvgPool(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int>& windowShapeArgs, const std::vector<int> & strideShape);
  PaddedAvgPool(LayerShapeL2Params & params) : PaddedAvgPool(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShapeArgs_, params.strideShapeArgs_) {};
};


int PaddedAvgPool::s_TotalNumber = 0;
const std::string PaddedAvgPool::s_LayerName = "PaddedAvgPool";

}   // namespace NetworkConfiguration

#endif
