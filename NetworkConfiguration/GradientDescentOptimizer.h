#ifndef GradientDescentOptimizer_h
#define GradientDescentOptimizer_h

#include "Layer.h"
#include "LossFunction.h"

// TODO: after refactoring all layers, remove this
#include "FullyConnectedLayer.h"
#include "MeanSquaredLossFunction.h"

namespace NetworkConfiguration {

class GradientDescentOptimizer 
{
  tensorflow::Scope & m_Scope;
  tensorflow::Output m_LearningRate;

  virtual void applyGradient(const tensorflow::Input & variable, const tensorflow::Input & gradient);

public:
  GradientDescentOptimizer(tensorflow::Scope & scope, float learningRate);
  // propagates error through whole network and updates parameters
  void propagate(std::vector<LayerP> & network, LossFunctionP lossFunctionPtr);
};

typedef std::shared_ptr<GradientDescentOptimizer> OptimizerP;
typedef std::function<GradientDescentOptimizer*(tensorflow::Scope &, float)> OptimizerCreator;
typedef Common::Factory<GradientDescentOptimizer, std::string, OptimizerCreator> OptimizerFactory;

} // namespace NetworkConfiguration

#endif
