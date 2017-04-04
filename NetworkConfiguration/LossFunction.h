#ifndef LossFunction_h
#define LossFunction_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

namespace NetworkConfiguration {

class LossFunction
{
public:
	virtual const tensorflow::Output& getLossOutput() const = 0;
	virtual Shape outputShape() const = 0;
	virtual ~LossFunction() {}

};	

typedef std::shared_ptr<LossFunction> LossFunctionP;

}// namespace NetworkConfiguration


#endif
