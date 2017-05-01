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
  PaddedAvgPool(tensorflow::Scope &scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const Shape& windowShape, const Shape & strideShape);
  PaddedAvgPool(LayerShapeL2Params & params) : PaddedAvgPool(params.scope_, params.previousLayerOutput_, params.previousLayerOutputShape_, params.paramShape_, params.strideShape_) {};
};


int PaddedAvgPool::s_TotalNumber = 0;
const std::string PaddedAvgPool::s_LayerName = "PaddedAvgPool";

}   // namespace NetworkConfiguration

#endif
