#include "AlgBackpropagation.h"
#include "NetworkConfiguration/AdamOptimizer.h"

void Backpropagation::nextIteration(const int & currGeneration)
{
  m_pEvalOp->setBatch(currGeneration);
  m_pOptimizer->advanceIteration();
}

Backpropagation::Backpropagation() : m_pSession(NewSession({}))
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
}

bool Backpropagation::initialize(StateP state)
{
  m_InitialLearningRate = *static_cast<float *>(getParameterValue(state, "initialLearningRate").get());
  m_FinalLearningRate = *static_cast<float *>(getParameterValue(state, "finalLearningRate").get());
  m_NumSteps = *static_cast<unsigned int *>(getParameterValue(state, "numSteps").get());
  m_WeightDecay = *static_cast<float *>(getParameterValue(state, "weightDecay").get());
  m_OptimizerName = *static_cast<std::string*>(getParameterValue(state, "optimizer").get());
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
    }
    catch (std::exception & e)
    {
      ECF_LOG_ERROR(state, e.what());
      return false;
    }
    // all done, ready to go
  }
  for (auto it = deme->begin(); it != deme->end(); ++it)
  {
    IndividualP individual = static_cast<IndividualP>(*it);
    // create input tensors from individual
    std::vector<std::pair<std::string, Tensor>> inputs = m_pEvalOp->createTensorsFromGenotype(individual);
    // set batch so the same batch is used in training and evaluation 
    nextIteration(state->getGenerationNo());
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
  }
  return true;
}
