#ifndef LossFunction_h
#define LossFunction_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <common/Shape.h>
#include <common/Factory.h>

namespace NetworkConfiguration {
  
struct LossBaseParams
{
  tensorflow::Scope & scope_;
  const tensorflow::Input & networkOutput_;
  const Shape & networkOutputShape_;
  const tensorflow::Input & expectedOutputsPlaceholder_;
  const Shape & expectedOutputShape_;
  std::string placeholderName_;
  LossBaseParams(tensorflow::Scope & scope, const tensorflow::Input & networkOutput, const Shape & networkOutputShape, const tensorflow::Input & expectedOutputsPlaceholder, const Shape & expectedOutputShape, std::string placeholderName) :
     scope_(scope), networkOutput_(networkOutput), networkOutputShape_(networkOutputShape), expectedOutputsPlaceholder_(expectedOutputsPlaceholder), expectedOutputShape_(expectedOutputShape), placeholderName_(placeholderName) {};
};


class LossFunction
{
protected:
  tensorflow::Output m_Loss;
  tensorflow::Scope & m_Scope;
  LossFunction(tensorflow::Scope &scope) : m_Scope(scope) {};
  LossFunction(LossBaseParams params) : m_Scope(params.scope_) {};

public:
  virtual const tensorflow::Output& getLossOutput() { return m_Loss; };
  virtual ~LossFunction() = default;

};	

typedef std::shared_ptr<LossFunction> LossFunctionP;
typedef std::function<LossFunction*(LossBaseParams &)> LossCreator;
typedef Common::Factory<LossFunction, std::string, LossCreator> LossFactory;
}// namespace NetworkConfiguration


#endif
