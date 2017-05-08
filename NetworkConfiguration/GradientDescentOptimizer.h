#ifndef GradientDescentOptimizer_h
#define GradientDescentOptimizer_h

#include "Layer.h"
#include "LossFunction.h"

namespace NetworkConfiguration {

class GradientDescentOptimizer 
{
  tensorflow::Scope & m_Scope;
  tensorflow::Output m_LearningRate;

  virtual void applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient);

public:
  GradientDescentOptimizer(tensorflow::Scope & scope, float learningRate);
  // propagates error through whole network and updates parameters, returns collection of names through which to fetch new values
  std::vector<std::string> propagate(const std::vector<LayerP> & network, LossFunctionP lossFunctionPtr);
};

typedef std::shared_ptr<GradientDescentOptimizer> OptimizerP;
typedef std::function<GradientDescentOptimizer*(tensorflow::Scope &, float)> OptimizerCreator;
typedef Common::Factory<GradientDescentOptimizer, std::string, OptimizerCreator> OptimizerFactory;

} // namespace NetworkConfiguration

#endif
