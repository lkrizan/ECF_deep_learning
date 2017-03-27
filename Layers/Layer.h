#ifndef Layer_h
#define Layer_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

#define INPUTS_PLACEHOLDER_NAME "inputs"
#define OUTPUTS_PLACEHOLDER_NAME "outputs"
#define LOSS_OUTPUT_NAME "loss"

using Layers::Shape;

class Layer
{
public:
	virtual const tensorflow::Output& forward() const = 0;
	virtual Shape outputShape() const = 0;
	virtual bool hasParams() const = 0;
};

class ParameterizedLayer : public Layer
{
public:
	bool hasParams() const override { return true; };
	// returns shapes of all parameters
	virtual std::vector<std::pair<std::string, Shape>> getParamShapes() const = 0;
};

class NonParameterizedLayer : public Layer
{
public:
	bool hasParams() const override { return false; };
};

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

#endif