#ifndef LossLayer_h
#define LossLayer_h

#include "NonParameterizedLayer.h"

namespace Layers {

class LossLayer : public NonParameterizedLayer
{
private:
	static int s_NumLossLayers;
protected:
	LossLayer() 
	{ 
		if (++s_NumLossLayers > 1)
			throw std::logic_error("Only one loss layer can be defined.\n");
	}
};

int LossLayer::s_NumLossLayers = 0;

}	// namespace Layers


#endif
