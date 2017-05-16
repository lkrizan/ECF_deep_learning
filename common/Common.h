#ifndef Common_h
#define Common_h

#include "Factory.h"
#include <tensorflow/core/framework/tensor.h>

namespace Common {

template<class T, class InputIterator>
void setTensor(tensorflow::Tensor &tensor, InputIterator first, InputIterator last)
{
  auto tensorMap = tensor.flat<T>();
  std::copy(first, last, tensorMap.data());
}

}   // namespace Common

#endif