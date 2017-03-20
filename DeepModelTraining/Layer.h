#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

class Layer
{
public:
	virtual tensorflow::Output forward() const = 0;
	// TODO: switch this method to Layer::TensorShape
	virtual const std::vector<int>& outputShape() = 0;
};

class ParameterizedLayer : public Layer
{
public:
	bool hasParams() { return true; };
	virtual const std::vector<std::pair<std::string, std::vector<int>>>& getParamShapes() = 0;
};

class NonParameterizedLayer : public Layer
{
public:
	bool hasParams() { return false; };
};
