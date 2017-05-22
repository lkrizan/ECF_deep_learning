#ifndef AlgBackpropagation_h
#define AlgBackpropagation_h

#include <ECF/ECF.h>
#include <ECF/Algorithm.h>
#include <DeepModelTraining/ModelEvalOp.h>
#include <NetworkConfiguration/Layer.h>
#include <NetworkConfiguration/LossFunction.h>
#include <NetworkConfiguration/GradientDescentOptimizer.h>

/*
  Backpropagation algorithm for FloatingPoint genotype, works only with ModelEvalOp evaluation operator
  (deep learning evaluation operator).
*/

class Backpropagation : public Algorithm
{
  std::string m_OptimizerName;
  // new session which will be used for backpropagation only
  std::unique_ptr<Session> m_pSession;
  float m_InitialLearningRate;
  float m_FinalLearningRate;
  unsigned int m_NumSteps;
  float m_WeightDecay;
  NetworkConfiguration::OptimizerP m_pOptimizer;
  bool m_Initialized = false;
  // name of variables that need to be fetched from tensorflow session
  std::vector<std::string> m_Variables;
  // all values to fetch
  std::vector<std::string> m_AllFetchValues;
  // pointer to evaluation operator
  ModelEvalOpP m_pEvalOp;
  // pointer to the nested algorithm (for hybrid algorithms)
  AlgorithmP m_pAlgorithm;

  // helper function which is used to fetch new batch and assign optimizer's iteration counter
  void nextIteration(const int & currGeneration);

public:
  Backpropagation();
  ~Backpropagation();
  void registerParameters(StateP state) override;
  bool initialize(StateP state) override;
  bool advanceGeneration(StateP state, DemeP deme) override;
};

typedef boost::shared_ptr<Backpropagation> BackpropagationP;

#endif