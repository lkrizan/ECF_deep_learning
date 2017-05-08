#include "AlgBackpropagation.h"

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
  // for now, this method does nothing
  // will be registering learning rate and weight decay as params
}

bool Backpropagation::initialize(StateP state)
{
  // also, does nothing for now, because algorithms are initialized before evaluation operator
  // TODO: ensure that size of population is 1
  return true;
}

bool Backpropagation::advanceGeneration(StateP state, DemeP deme)
{
  // algorithms are initialized before evaluation operators, so this step must be done here
  if (!m_Initialized)
  {
    m_Initialized = true;
    try
    {
      m_pEvalOp = dynamic_pointer_cast<ModelEvalOp>(state->getEvalOp());
    }
    catch(std::exception & e)
    {
      ECF_LOG_ERROR(state, e.what());
      return false;
    }
    ECF_LOG(state, 4, "Creating graph definition...");
    // create graph definition for backpropagation
    Scope & scope = m_pEvalOp->getScope();
    // hardcoded learning rate for now
    m_pOptimizer = NetworkConfiguration::OptimizerP(new NetworkConfiguration::GradientDescentOptimizer(scope, m_LearningRate));
    m_Variables = m_pOptimizer->propagate(m_pEvalOp->getNetwork(), m_pEvalOp->getLossFunction());
    ECF_LOG(state, 4, "Creating session...");
    GraphDef gdef;
    TF_CHECK_OK(scope.ToGraphDef(&gdef));
    Status status = m_pSession->Create(gdef);
    ECF_LOG(state, 5, "Session creation status:");
    ECF_LOG(state, 5, status.ToString());
    ECF_LOG(state, 5, "Graph definition data:");
    ECF_LOG(state, 5, gdef.DebugString());
    // all done, ready to go
  }
  IndividualP individual = static_cast<IndividualP>(deme->at(0));
  // create input tensors from individual
  std::vector<std::pair<std::string, Tensor>> inputs = m_pEvalOp->createTensorsFromGenotype(individual);
  // set batch so the same batch is used in training and evaluation 
  m_pEvalOp->setBatch(state->getGenerationNo());
  inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, m_pEvalOp->getCurrentInputs()));
  inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, m_pEvalOp->getCurrentOutputs()));
  std::vector<Tensor> outputs;
  // run session and get new parameter values
  Status status = m_pSession->Run(inputs, m_Variables, {}, &outputs);
  // clear old genotype values
  FloatingPoint::FloatingPoint* gen = static_cast<FloatingPoint::FloatingPoint*>(individual->getGenotype().get());
  std::vector<double> & data = gen->realValue;
  data.clear();
  // copy new parameter values to the genotype
  for_each(outputs.begin(), outputs.end(), [&data](const Tensor & tensor)
    {
      auto size = tensor.flat<float>().size();
      std::copy(tensor.flat<float>().data(), tensor.flat<float>().data() + size, std::back_inserter(data));
    });
  // evaluate new individual
  evaluate(individual);
  return true;
}
