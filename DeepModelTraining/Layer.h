#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

class Layer
{
public:
	virtual const tensorflow::Output& forward() const = 0;
	// TODO: switch this method to Layer::TensorShape
	virtual Shape outputShape() = 0;
	virtual bool hasParams() = 0;
};

class ParameterizedLayer : public Layer
{
public:
	bool hasParams() override { return true; };
	// returns shapes of all parameters
	virtual std::vector<std::pair<std::string, Shape>> getParamShapes() const = 0;
};

class NonParameterizedLayer : public Layer
{
public:
	bool hasParams() override { return false; };
};
