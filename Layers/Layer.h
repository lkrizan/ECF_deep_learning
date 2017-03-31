#ifndef Layer_h
#define Layer_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

#define INPUTS_PLACEHOLDER_NAME "inputs"
#define OUTPUTS_PLACEHOLDER_NAME "outputs"
#define LOSS_OUTPUT_NAME "loss"

namespace Layers {

class Layer
{
public:
	virtual const tensorflow::Output& forward() const = 0;
	virtual Shape outputShape() const = 0;
	virtual bool hasParams() const = 0;
	virtual ~Layer() {}
};

typedef std::shared_ptr<Layer> LayerP;

}
#endif