#include "AlgBackpropagation.h"
#include "NetworkConfiguration/AdamOptimizer.h"
#include "AlgMicrocanonicalAnnealing.h"

void Backpropagation::nextIteration(const int & currGeneration)
{
  m_pEvalOp->setBatch(currGeneration);
  m_pOptimizer->advanceIteration();
}

// TODO: rethink this method (not used anywhere for now)
void Backpropagation::reinitializePopulation(DemeP deme, uint numberOfMutations)
{
  IndividualP source = deme->at(0);
  for (uint idx = 1; idx < deme->size(); ++idx)
  {
    replaceWith(idx, IndividualP(source->copy()));
    // perform mutation several times to achive greater difference between individuals
    for (int i = 0; i < numberOfMutations; ++i)
      mutate(deme->at(idx));
  }
}

Backpropagation::Backpropagation() : m_pSession(NewSession({})), m_SelBestOp(new SelBestOp)
{
  name_ = "Backpropagation";
}

Backpropagation::~Backpropagation()
{
  m_pSession->Close();
}

void Backpropagation::registerParameters(StateP state)
{
  registerParameter(state, "initialLearningRate", (voidP) new float(1e-5), ECF::FLOAT);
  registerParameter(state, "finalLearningRate", (voidP) new float(1e-5), ECF::FLOAT);
  registerParameter(state, "numSteps", (voidP) new unsigned int(1), ECF::UINT);
  registerParameter(state, "weightDecay", (voidP) new float(1e-4), ECF::FLOAT);
  registerParameter(state, "optimizer", (voidP) new std::string("GradientDescentOptimizer"), ECF::STRING);
  registerParameter(state, "nestedAlgorithm", (voidP) new std::string(""), ECF::STRING);
  registerParameter(state, "nestedAlgorithmGenerations", (voidP) new int(10), ECF::INT);

  // TODO: register all ECF algorithms in AlgorithmFactory
  AlgorithmFactory::instance().registerClass("GeneticAnnealing", []() {return new GeneticAnnealing;});
  AlgorithmFactory::instance().registerClass("SteadyStateTournament", []() {return new SteadyStateTournament;});
}

bool Backpropagation::initialize(StateP state)
{
  m_InitialLearningRate = *static_cast<float *>(getParameterValue(state, "initialLearningRate").get());
  m_FinalLearningRate = *static_cast<float *>(getParameterValue(state, "finalLearningRate").get());
  m_NumSteps = *static_cast<unsigned int *>(getParameterValue(state, "numSteps").get());
  m_WeightDecay = *static_cast<float *>(getParameterValue(state, "weightDecay").get());
  m_OptimizerName = *static_cast<std::string*>(getParameterValue(state, "optimizer").get());

  m_SelBestOp->initialize(state);
  m_NestedAlgorithmName = *static_cast<std::string*>(getParameterValue(state, "nestedAlgorithm").get());
  m_NestedAlgorithmGenerations = *static_cast<int*>(getParameterValue(state, "nestedAlgorithmGenerations").get());

  if (m_NestedAlgorithmName.length() != 0)
    m_UseNestedAlgorithm = true;
  return true;
}

bool Backpropagation::advanceGeneration(StateP state, DemeP deme)
{
  // algorithms are initialized before evaluation operators, so this step must be done here
  if (!m_Initialized)
  {
    m_Initialized = true;
    m_pEvalOp = dynamic_pointer_cast<ModelEvalOp>(state->getEvalOp());
    if (m_pEvalOp == nullptr)
    {
      ECF_LOG_ERROR(state, "Backpropagation algorithm can only be used with ModelEvalOp evaluation operator.");
      return false;
    }
    ECF_LOG(state, 4, "Creating graph definition...");
    // create graph definition for backpropagation
    try
    {
      Scope & scope = m_pEvalOp->getScope();
      m_pOptimizer = NetworkConfiguration::OptimizerFactory::instance().createObject(m_OptimizerName, NetworkConfiguration::OptimizerParams(scope, m_InitialLearningRate, m_FinalLearningRate, m_NumSteps, m_WeightDecay));
      m_Variables = m_pOptimizer->propagate(m_pEvalOp->getNetwork(), m_pEvalOp->getLossFunction());
      std::vector<std::string> fetchVals = m_pOptimizer->getFetchList();
      m_AllFetchValues.reserve(m_Variables.size() + fetchVals.size());
      m_AllFetchValues.insert(m_AllFetchValues.end(), m_Variables.begin(), m_Variables.end());
      m_AllFetchValues.insert(m_AllFetchValues.end(), fetchVals.begin(), fetchVals.end());
      ECF_LOG(state, 4, "Creating session...");
      GraphDef gdef;
      TF_CHECK_OK(scope.ToGraphDef(&gdef));
      Status status = m_pSession->Create(gdef);
      if (!status.ok())
      {
        ECF_LOG_ERROR(state, "Session creation failed:\n");
        ECF_LOG_ERROR(state, status.ToString());
      }
      ECF_LOG(state, 4, "Graph definition data:");
      ECF_LOG(state, 4, gdef.DebugString());

      if (m_UseNestedAlgorithm)
      {
        m_pNestedAlgorithm = AlgorithmFactory::instance().createObject(m_NestedAlgorithmName);
        // set genetic operators and initialize algorithm
        m_pNestedAlgorithm->mutation_ = mutation_;
        m_pNestedAlgorithm->crossover_ = crossover_;
        m_pNestedAlgorithm->evalOp_ = evalOp_;
        m_pNestedAlgorithm->state_ = state_;
        m_pNestedAlgorithm->initialize(state);
      }
    }
    catch (std::exception & e)
    {
      ECF_LOG_ERROR(state, e.what());
      throw e;
    }
  }
  nextIteration(state->getGenerationNo());
  IndividualP individual = static_cast<IndividualP>(deme->at(0));
  // create input tensors from individual
  std::vector<std::pair<std::string, Tensor>> inputs = m_pEvalOp->createTensorsFromGenotype(individual);
  // set batch so the same batch is used in training and evaluation 
  inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, m_pEvalOp->getCurrentInputs()));
  inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, m_pEvalOp->getCurrentOutputs()));
  // workaround for passing gradient momentum
  std::vector<std::pair<std::string, Tensor>> momentumFeedValues = m_pOptimizer->getFeedList();
  inputs.insert(inputs.end(), momentumFeedValues.begin(), momentumFeedValues.end());
  // run session and get new parameter values
  std::vector<Tensor> outputs;
  Status status = m_pSession->Run(inputs, m_AllFetchValues, {}, &outputs);
  // clear old genotype values
  FloatingPoint::FloatingPoint* gen = static_cast<FloatingPoint::FloatingPoint*>(individual->getGenotype().get());
  std::vector<double> & data = gen->realValue;
  data.clear();
  // copy new parameter values to the genotype
  for_each(outputs.begin(), outputs.begin() + m_Variables.size(), [&data](const Tensor & tensor)
  {
    auto size = tensor.flat<float>().size();
    std::copy(tensor.flat<float>().data(), tensor.flat<float>().data() + size, std::back_inserter(data));
  });
  // workaround for setting variables' gradient momentum (if used)
  m_pOptimizer->setFeedList(std::vector<tensorflow::Tensor>(outputs.begin() + m_Variables.size(), outputs.end()));
  // evaluate new individual
  evaluate(individual);
  // run nested algorithm (for hybrid algorithms)
  if (m_UseNestedAlgorithm)
  {
    ECF_LOG(state, 4, "Running nested algorithm...");
    // reinitializePopulation(deme);
    for (uint i = 0; i < m_NestedAlgorithmGenerations; ++i)
      m_pNestedAlgorithm->advanceGeneration(state);
    // if population algorithm was used, select best individual for use with backpropagation (backprop uses individual at index 0)
    if (deme->size() > 1)
    {
      IndividualP best = m_SelBestOp->select(*deme);
      if (best != deme->at(0))
      {
        std::swap(deme->at(0)->index, best->index);
        std::swap(deme->at(0), best);
      }
    }
  }
  return true;
}
