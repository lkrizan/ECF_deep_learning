#ifndef Layer_h
#define Layer_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "Shape.h"

namespace NetworkConfiguration {

class Layer
{
public:
  virtual const tensorflow::Output& forward() const = 0;
  virtual Shape outputShape() const = 0;
  virtual bool hasParams() const = 0;
  virtual ~Layer() = default;
};

typedef std::shared_ptr<Layer> LayerP;

}	// namespace NetworkConfiguration
#endif