#ifndef GradientDescentOptimizer_h
#define GradientDescentOptimizer_h

#include "Layer.h"
#include "LossFunction.h"

#define POSTFIX "final"
#define LEARNING_RATE_STR "learning_rate"

namespace NetworkConfiguration {

struct OptimizerParams
{
  tensorflow::Scope & scope_;
  float initialLearningRate_;
  float finalLearningRate_;
  unsigned int numSteps_;
  float weightDecay_;
  OptimizerParams(tensorflow::Scope & scope, float initialLearningRate, float finalLearningRate, unsigned int numSteps, float weightDecay) : 
    scope_(scope), initialLearningRate_(initialLearningRate), finalLearningRate_(finalLearningRate), numSteps_(numSteps), weightDecay_(weightDecay) {};
};

class GradientDescentOptimizer 
{
protected:
  tensorflow::Scope & m_Scope;
  /*
  Applies learning rate by formula:
    lr_k = (1 - alpha) * lr_0 + alpha * lr_t,
  where k is current iteration, lr_0 initial learning rate, and lr_t final learning rate
  (alpha is k/t, lr_t is used later on until the algorithm ends)s
  */
  float m_InitialLearningRate;
  float m_FinalLearningRate;
  float m_CurrentLearningRate;
  unsigned int m_NumSteps;
  unsigned int m_CurrentIteration = 0;
  // tensorflow constant used for regularization
  tensorflow::Output m_WeightDecay;
  // placeholder for learning rate in current iteration
  tensorflow::ops::Placeholder m_LearningRatePlaceholder;

  virtual void applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient, const Shape & variableShape, const std::string & variableName);
  tensorflow::Output regularizationGradient(const tensorflow::Input & variable);
  // calculating new learning rate
  void adjustLearningRate();

public:
  virtual ~GradientDescentOptimizer() = default;
  GradientDescentOptimizer(tensorflow::Scope & scope, float initialLearningRate, float finalLearningRate, unsigned int numSteps,  float weightDecay);
  GradientDescentOptimizer(const OptimizerParams & params) : GradientDescentOptimizer(params.scope_, params.initialLearningRate_, params.finalLearningRate_, params.numSteps_, params.weightDecay_) {};
  // propagates error through whole network and updates parameters, returns collection of names through which to fetch new values
  std::vector<std::string> propagate(const std::vector<LayerP> & network, LossFunctionP lossFunctionPtr);
  /*
  At the time of writing this, tensorflow's documentation about using Variable class does not exist,
  so this is a workaround for applying momentum for e.g., Adam, Adagrad, or similar optimizers.
  For basic gradient descent optimizers, these functions mostly do nothing.
  */
  std::vector<std::pair<std::string, tensorflow::Tensor>> getFeedList();
  virtual void setFeedList(std::vector<tensorflow::Tensor> & tensors) {};
  virtual std::vector<std::string> getFetchList() { return std::vector<std::string>(); }
  virtual void advanceIteration() { ++m_CurrentIteration; adjustLearningRate(); }

protected:
  // actual step for returning specialized feed list - for regular gradient descent it does nothing 
  virtual std::vector<std::pair<std::string, tensorflow::Tensor>> doGetFeedList() { return std::vector<std::pair<std::string, tensorflow::Tensor>>(); }
};

typedef std::shared_ptr<GradientDescentOptimizer> OptimizerP;
typedef std::function<GradientDescentOptimizer*(OptimizerParams &)> OptimizerCreator;
typedef Common::Factory<GradientDescentOptimizer, std::string, OptimizerCreator> OptimizerFactory;

} // namespace NetworkConfiguration

#endif
