#ifndef GradientDescentOptimizer_h
#define GradientDescentOptimizer_h

#include "Layer.h"
#include "LossFunction.h"

namespace NetworkConfiguration {

struct OptimizerParams
{
  tensorflow::Scope & scope_;
  float learningRate_;
  float weightDecay_;
  OptimizerParams(tensorflow::Scope & scope, float learningRate, float weightDecay) : scope_(scope), learningRate_(learningRate), weightDecay_(weightDecay) {};
};

class GradientDescentOptimizer 
{
  tensorflow::Scope & m_Scope;
  tensorflow::Output m_LearningRate;
  tensorflow::Output m_WeightDecay;

  virtual void applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient);
  tensorflow::Output regularizationGradient(const tensorflow::Input & variable);

public:
  GradientDescentOptimizer(tensorflow::Scope & scope, float learningRate, float weightDecay);
  GradientDescentOptimizer(const OptimizerParams & params) : GradientDescentOptimizer(params.scope_, params.learningRate_, params.weightDecay_) {};
  // propagates error through whole network and updates parameters, returns collection of names through which to fetch new values
  std::vector<std::string> propagate(const std::vector<LayerP> & network, LossFunctionP lossFunctionPtr);
  /*
  At the time of writing this, tensorflow's documentation about using Variable does not exist,
  so this is a workaround for applying momentum for e.g., Adam, Adagrad, or similar optimizers.
  For basic gradient descent optimizers, these functions do absolutely nothing.
  */
  virtual std::vector<std::pair<std::string, tensorflow::Tensor>> getFeedList() { return std::vector<std::pair<std::string, tensorflow::Tensor>>(); }
  virtual void setFeedList(std::vector<tensorflow::Tensor> & tensors) {};
};

typedef std::shared_ptr<GradientDescentOptimizer> OptimizerP;
typedef std::function<GradientDescentOptimizer*(OptimizerParams &)> OptimizerCreator;
typedef Common::Factory<GradientDescentOptimizer, std::string, OptimizerCreator> OptimizerFactory;

} // namespace NetworkConfiguration

#endif
