#include "GradientDescentOptimizer.h"

namespace NetworkConfiguration {
  
GradientDescentOptimizer::GradientDescentOptimizer(tensorflow::Scope & scope, float learningRate) : m_Scope(scope)
{
  m_LearningRate = tensorflow::ops::Const(m_Scope, learningRate);
}

void GradientDescentOptimizer::applyGradient(const tensorflow::Input & variable, const tensorflow::Input & gradient)
{
  using namespace tensorflow::ops;
  auto result = ApplyGradientDescent(m_Scope, variable, m_LearningRate, gradient);
}

void GradientDescentOptimizer::propagate(std::vector<LayerP> & network, LossFunctionP lossFunctionPtr)
{
  // calculate gradient through loss function
  auto gradient = lossFunctionPtr->backward();
  
  // propagate gradient through the network in reverse order
  for (auto it = network.rbegin(); it != network.rend(); ++it)
  {
    if (!(*it)->hasParams())
    {
      // layer has no parameters, so just calculate gradient over inputs
      gradient = (*it)->backwardInputs(gradient);
      continue;
    }
    else
    {
      auto layerPtr = std::dynamic_pointer_cast<ParameterizedLayer>(*it);
      // apply gradient over parameters
      auto gradWeights = layerPtr->backwardWeights(gradient);
      applyGradient(layerPtr->getWeights(), gradWeights);
      auto gradBias = layerPtr->backwardBias(gradient);
      applyGradient(layerPtr->getBias(), gradBias);
      // calculate gradient over inputs
      gradient = layerPtr->backwardInputs(gradient);
    }
  }
}

}   // namespace NetworkConfiguration

// register class in factory
namespace {
  using namespace NetworkConfiguration;
  OptimizerCreator ctor = [](tensorflow::Scope & scope, float learningRate) {return new GradientDescentOptimizer(scope, learningRate);};
  bool dummy = OptimizerFactory::instance().registerClass("GradientDescentOptimizer", ctor);
}
