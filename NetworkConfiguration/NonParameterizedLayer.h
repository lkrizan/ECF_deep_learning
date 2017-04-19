#ifndef NonParameterizedLayer_h
#define NonParameterizedLayer_h

#include "Layer.h"

namespace NetworkConfiguration {

class NonParameterizedLayer : public Layer
{
public:
  virtual ~NonParameterizedLayer() = default;
  bool hasParams() const override { return false; };
};

typedef std::shared_ptr<NonParameterizedLayer> NonParameterizedLayerP;

} // namespace NetworkConfiguration

#endif
