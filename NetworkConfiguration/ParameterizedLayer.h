#ifndef ParameterizedLayer_h
#define ParameterizedLayer_h

#include "Layer.h"

namespace NetworkConfiguration {

class ParameterizedLayer : public Layer
{
public:
  virtual ~ParameterizedLayer() = default;
  bool hasParams() const override { return true; };
  // returns shapes of all parameters
  virtual std::vector<std::pair<std::string, Shape>> getParamShapes() const = 0;
};

typedef std::shared_ptr<ParameterizedLayer> ParameterizedLayerP;

} // namespace NetworkConfiguration

#endif
