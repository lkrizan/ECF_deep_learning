#include "AdamOptimizer.h"

namespace NetworkConfiguration {

void AdamOptimizer::applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient, const Shape & variableShape, const std::string & variableName)
{
  using namespace tensorflow::ops;
  // constant used throughout this method
  auto one = Const(m_Scope, 1.f);
  // create placeholders form moment estimates
  const std::string firstMomentName = variableName + FIRST_MOMENT_ESTIMATE;
  const std::string secondMomentName = variableName + SECOND_MOMENT_ESTIMATE;
  auto s = Placeholder(m_Scope.WithOpName(firstMomentName), tensorflow::DT_FLOAT);
  auto r = Placeholder(m_Scope.WithOpName(secondMomentName), tensorflow::DT_FLOAT);
  // initial values for tensors holding moment (conveniently, this vector constructor fills the vector with zeros)
  std::vector<int> vals(variableShape.numberOfElements());
  // both moment estimates use the same initial value, so one value is enough
  tensorflow::Tensor moment(tensorflow::DT_FLOAT, variableShape.asTensorShape());
  Common::setTensor<float>(moment, vals.begin(), vals.end());
  // set the values in future feed list
  m_GradientMomentum.push_back(std::make_pair(firstMomentName, moment));
  m_GradientMomentum.push_back(std::make_pair(secondMomentName, moment));
  // set names for future fetch list
  const std::string firstMomentFetchName = firstMomentName + POSTFIX;
  const std::string secondMomentFetchName = secondMomentName + POSTFIX;
  m_FetchList.push_back(firstMomentFetchName);
  m_FetchList.push_back(secondMomentFetchName);
  // first moment estimate calculation
  auto fMomEstTemp1 = Multiply(m_Scope, m_Rho1, s);
  auto fMomEstTemp2 = Multiply(m_Scope, m_Rho1Inv, gradient);
  auto sNew = Add(m_Scope.WithOpName(firstMomentFetchName), fMomEstTemp1, fMomEstTemp2);
  // second moment estimate calculation
  auto sMomEstTemp1 = Multiply(m_Scope, m_Rho2, r);
  auto sMomEstTemp2 = Multiply(m_Scope, m_Rho2Inv, gradient);
  auto sMomEstTemp3 = Multiply(m_Scope, sMomEstTemp2, gradient);
  auto rNew = Add(m_Scope.WithOpName(secondMomentFetchName), sMomEstTemp1, sMomEstTemp3);
  // correct bias in first moment
  auto fMomNominator = Subtract(m_Scope, one, Pow(m_Scope, m_Rho1, m_Iteration));
  auto sFinal = Div(m_Scope, sNew, fMomNominator);
  // correct bias in second moment
  auto sMomNominator = Subtract(m_Scope, one, Pow(m_Scope, m_Rho2, m_Iteration));
  auto rFinal = Div(m_Scope, rNew, sMomNominator);
  // corrected gradient value
  auto gradientNew = Div(m_Scope, sFinal, Add(m_Scope, Sqrt(m_Scope, rNew), m_Delta));
  // finally, apply the gradient
  auto temp = Multiply(m_Scope, gradientNew, m_LearningRatePlaceholder);
  auto result = Subtract(m_Scope.WithOpName(name), variable, temp);
}


AdamOptimizer::AdamOptimizer(tensorflow::Scope & scope, float initialLearningRate, float finalLearningRate, unsigned int numSteps, float weightDecay, float rho1, float rho2) :
  GradientDescentOptimizer(scope, initialLearningRate, finalLearningRate, numSteps, weightDecay)
{
  m_Rho1 = tensorflow::ops::Const(m_Scope, rho1);
  m_Rho1Inv = tensorflow::ops::Const(m_Scope, 1 - rho1);
  m_Rho2 = tensorflow::ops::Const(m_Scope, rho2);
  m_Rho2Inv = tensorflow::ops::Const(m_Scope, 1 - rho2);
  m_Iteration = tensorflow::ops::Placeholder(m_Scope.WithOpName(CURR_ITER), tensorflow::DT_FLOAT);
  m_Delta = tensorflow::ops::Const(m_Scope, (float)1e-8);
}

std::vector<std::pair<std::string, tensorflow::Tensor>> AdamOptimizer::doGetFeedList()
{
  std::vector<std::pair<std::string, tensorflow::Tensor>> feedList = m_GradientMomentum;
  // set current iteration
  tensorflow::Tensor currIter(tensorflow::DT_FLOAT, Shape({ 1 }).asTensorShape());
  currIter.flat<float>()(0) = static_cast<float>(m_CurrentIteration);
  feedList.push_back(std::make_pair(CURR_ITER, currIter));
  return feedList;
}

void AdamOptimizer::setFeedList(std::vector<tensorflow::Tensor>& tensors)
{
  for (unsigned int i = 0; i < m_GradientMomentum.size(); ++i)
  {
    m_GradientMomentum[i].second = tensors[i];
  }
}

std::vector<std::string> AdamOptimizer::getFetchList()
{
  return m_FetchList;
}

}   // namespace NetworkConfiguration

// register class in factory
namespace {
  using namespace NetworkConfiguration;
  OptimizerCreator ctor = [](OptimizerParams & params) {return new AdamOptimizer(params);};
  bool dummy = OptimizerFactory::instance().registerClass("AdamOptimizer", ctor);
}