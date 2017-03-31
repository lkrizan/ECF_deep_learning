#ifndef NonParameterizedLayer_h
#define NonParameterizedLayer_h

#include "Layer.h"

namespace Layers {

class NonParameterizedLayer : public Layer
{
public:
	bool hasParams() const override { return false; };
};

typedef std::shared_ptr<NonParameterizedLayer> NonParameterizedLayerP;

} // namespace Layers

#endif
