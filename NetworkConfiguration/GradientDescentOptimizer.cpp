#include "GradientDescentOptimizer.h"
#include <algorithm>

namespace NetworkConfiguration {
  
GradientDescentOptimizer::GradientDescentOptimizer(tensorflow::Scope & scope, float initialLearningRate, float finalLearningRate, unsigned int numSteps, float weightDecay) 
  : m_Scope(scope), m_LearningRatePlaceholder(m_Scope.WithOpName(LEARNING_RATE_STR), tensorflow::DT_FLOAT)
{
  m_InitialLearningRate = initialLearningRate;
  m_FinalLearningRate = finalLearningRate;
  m_NumSteps = numSteps;
  m_WeightDecay = tensorflow::ops::Const(m_Scope, weightDecay);
}

void GradientDescentOptimizer::applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient, const Shape & variableShape, const std::string & variableName)
{
  using namespace tensorflow::ops;
  auto temp = Multiply(m_Scope, gradient, m_LearningRatePlaceholder);
  auto result = Subtract(m_Scope.WithOpName(name), variable, temp);
}

tensorflow::Output GradientDescentOptimizer::regularizationGradient(const tensorflow::Input & variable)
{
  using namespace tensorflow::ops;
  return Multiply(m_Scope, m_WeightDecay, variable);
}

void GradientDescentOptimizer::adjustLearningRate()
{
  if (m_CurrentIteration < m_NumSteps)
  {
    const float alpha = static_cast<float>(m_CurrentIteration) / m_NumSteps;
    m_CurrentLearningRate = (1.f - alpha) * m_InitialLearningRate + alpha * m_FinalLearningRate;
  }
  else
  {
    m_CurrentLearningRate = m_FinalLearningRate;
  }
}

std::vector<std::string> GradientDescentOptimizer::propagate(const std::vector<LayerP> & network, LossFunctionP lossFunctionPtr)
{
  // container for return value
  std::vector<std::string> variables;
  // final vector 100% sure will not be larger than this
  variables.reserve(2 * network.size());
  const std::string postfix = POSTFIX;
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
      const std::string layerName = layerPtr->layerName();
      const std::string wName = layerName + "_w" + postfix;
      const std::string bName = layerName + "_b" + postfix;
      // calculate gradient over inputs
      auto newGradient = layerPtr->backwardInputs(gradient);
      // apply gradient over parameters
      auto gradWeights = layerPtr->backwardWeights(gradient);
      // add regularization (weights only)
      auto gradTotal = tensorflow::ops::Add(m_Scope, gradWeights, regularizationGradient(layerPtr->getWeights()));
      applyGradient(wName, layerPtr->getWeights(), gradTotal, layerPtr->getWeightsShape(), layerName + "_w");
      auto gradBias = layerPtr->backwardBias(gradient);
      applyGradient(bName, layerPtr->getBias(), gradBias, layerPtr->getBiasShape(), layerName + "_b");
      gradient = newGradient;
      // append variable names
      variables.push_back(bName);
      variables.push_back(wName);
    }
  }
  // reverse the vector because layers were iterated backwards
  std::reverse(variables.begin(), variables.end());
  return variables;
}

std::vector<std::pair<std::string, tensorflow::Tensor>> GradientDescentOptimizer::getFeedList()
{
  {
    std::vector<std::pair<std::string, tensorflow::Tensor>> result;
    // add current learning rate to the feed list
    tensorflow::Tensor learningRateTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1 }));
    learningRateTensor.flat<float>()(0) = m_CurrentLearningRate;
    result.push_back(std::make_pair(LEARNING_RATE_STR, learningRateTensor));
    // append the remaining feed values to the vector (if there are any)
    std::vector<std::pair<std::string, tensorflow::Tensor>> additionalValues = this->doGetFeedList();
    result.insert(result.end(), additionalValues.begin(), additionalValues.end());
    return result;
  }
}

}   // namespace NetworkConfiguration

// register class in factory
namespace {
  using namespace NetworkConfiguration;
  OptimizerCreator ctor = [](OptimizerParams & params) {return new GradientDescentOptimizer(params);};
  bool dummy = OptimizerFactory::instance().registerClass("GradientDescentOptimizer", ctor);
}
