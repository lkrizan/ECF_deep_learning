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
  const tensorflow::Input & m_NetworkOutput;
  const tensorflow::Input & m_ExpectedOutputs;
  LossFunction(tensorflow::Scope &scope, const tensorflow::Input & networkOutput, const tensorflow::Input & expectedOutputsPlaceholder) : 
    m_Scope(scope), m_NetworkOutput(networkOutput), m_ExpectedOutputs(expectedOutputsPlaceholder) {};
  LossFunction(LossBaseParams params) : m_Scope(params.scope_), m_NetworkOutput(params.networkOutput_), m_ExpectedOutputs(params.expectedOutputsPlaceholder_) {};

public:
  virtual const tensorflow::Output& getLossOutput() { return m_Loss; };
  virtual ~LossFunction() = default;
  /*
  // returns gradient through loss function
  virtual const tensorflow::Output & backward() = 0;
  */

};	

typedef std::shared_ptr<LossFunction> LossFunctionP;
typedef std::function<LossFunction*(LossBaseParams &)> LossCreator;
typedef Common::Factory<LossFunction, std::string, LossCreator> LossFactory;
}// namespace NetworkConfiguration


#endif
