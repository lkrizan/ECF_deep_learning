#ifndef AdamOptimizer_h
#define AdamOptimizer_h

#include "GradientDescentOptimizer.h"
#include <common/Common.h>

#define CURR_ITER "curr_iter"
#define FIRST_MOMENT_ESTIMATE "_s"
#define SECOND_MOMENT_ESTIMATE "_r"

namespace NetworkConfiguration {

class AdamOptimizer : public GradientDescentOptimizer
{
  std::vector<std::pair<std::string, tensorflow::Tensor>> m_GradientMomentum;
  std::vector<std::string> m_FetchList;
  unsigned int m_CurrentIteration = 1;
  // exponential decay rates for moment estimates (and their 1 - rho variant)
  tensorflow::Output m_Rho1;
  tensorflow::Output m_Rho1Inv;
  tensorflow::Output m_Rho2;
  tensorflow::Output m_Rho2Inv;
  // placeholder for current timestep
  tensorflow::Output m_Iteration;
  // small constant used for numerical stabilization
  tensorflow::Output m_Delta;

protected:
  void applyGradient(const std::string & name, const tensorflow::Input & variable, const tensorflow::Input & gradient, const Shape & variableShape, const std::string & layerName) override;

public:
  AdamOptimizer(tensorflow::Scope & scope, float learningRate, float weightDecay, float rho1 = 0.9, float rho2 = 0.999);
  AdamOptimizer(const OptimizerParams & params) : AdamOptimizer(params.scope_, params.learningRate_, params.weightDecay_) {};
  std::vector<std::pair<std::string, tensorflow::Tensor>> getFeedList() override;
  void setFeedList(std::vector<tensorflow::Tensor> & tensors) override;
  std::vector<std::string> getFetchList() const override;
};

}

#endif
