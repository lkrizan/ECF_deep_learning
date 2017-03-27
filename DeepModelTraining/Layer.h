#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

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
