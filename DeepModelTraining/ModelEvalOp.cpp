#include "ModelEvalOp.h"
#include "ConfigParser.h"


void ModelEvalOp::registerParameters(StateP state)
{
  state->getRegistry()->registerEntry("configFilePath", (voidP)(new std::string), ECF::STRING);
  state->getRegistry()->registerEntry("saveModel", (voidP)(new int(0)), ECF::INT);
  state->getRegistry()->registerEntry("modelSavePath", (voidP)(new std::string), ECF::STRING);

}

ModelEvalOp::~ModelEvalOp()
{
  if (m_SaveModel)
  {
    try
    {
      saveDefinitionToFile();
    }
    catch (std::exception &e)
    {
      std::string errMsg = "Failed to save trained model: " + std::string(e.what());
      ECF_LOG_ERROR(m_ECFState, errMsg);
    }
  }

  m_pSession->Close();
}

void ModelEvalOp::saveDefinitionToFile() const
{
  ModelExporter exporter(m_ECFState, m_ModelExportPath);
  exporter.exportGraph(m_GraphDef);
  // get best individual from Hall of Fame
  IndividualP bestIndividual = m_ECFState->getHoF()->getBest().at(0);
  FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) bestIndividual->getGenotype().get();
  auto currentIterator = gen->realValue.begin();
  auto endIterator = gen->realValue.end();
  for_each(m_VariableData.begin(), m_VariableData.end(), [&exporter, &currentIterator](const VariableData & data) {exporter.exportVariableValues(data.m_VariableName, data.m_BasicShape, currentIterator, currentIterator + data.m_NumberOfElements); currentIterator += data.m_NumberOfElements;});
}


std::vector<NetworkConfiguration::LayerP> ModelEvalOp::createLayers(Scope &root, const std::vector<std::pair<std::string, std::vector<std::vector<int>>>>& networkConfiguration, const std::string lossFunctionName, const NetworkConfiguration::Shape & inputShape, const NetworkConfiguration::Shape & outputShape)
{
  using NetworkConfiguration::Shape;
  // create placeholders for inputs and for expected outputs so network can be defined
  m_InputsPlaceholder = ops::Placeholder(root.WithOpName(INPUTS_PLACEHOLDER_NAME), DT_FLOAT);
  m_OutputsPlaceholder = ops::Placeholder(root.WithOpName(OUTPUTS_PLACEHOLDER_NAME), DT_FLOAT);
  // create layers from value pairs - layer type name and shape in form of vector of ints
  std::vector<NetworkConfiguration::LayerP> layers;
  for (auto iter = networkConfiguration.begin(); iter != networkConfiguration.end(); iter++)
  {
    std::vector<int> paramShapeArgs = (iter->second.size() >= 1) ? iter->second.at(0) : std::vector<int>();
    std::vector<int> strideShapeArgs = (iter->second.size() >= 2) ? iter->second.at(1) : std::vector<int>();
    NetworkConfiguration::LayerP layer;
    if (iter == networkConfiguration.begin())
      layer = NetworkConfiguration::LayerFactory::instance().createObject(iter->first, NetworkConfiguration::LayerShapeL2Params(root, m_InputsPlaceholder, inputShape, paramShapeArgs, strideShapeArgs));
    else
      layer = NetworkConfiguration::LayerFactory::instance().createObject(iter->first, NetworkConfiguration::LayerShapeL2Params(root, layers.back()->forward(), layers.back()->outputShape(), paramShapeArgs, strideShapeArgs));
    layers.push_back(layer);
  }
  // add loss function to graph
  m_LossFunction = NetworkConfiguration::LossFactory::instance().createObject(lossFunctionName, NetworkConfiguration::LossBaseParams(root, layers.back()->forward(), layers.back()->outputShape(), m_OutputsPlaceholder, outputShape, LOSS_OUTPUT_NAME));
  return layers;
}

std::vector<ModelEvalOp::VariableData> ModelEvalOp::createVariableData(const std::vector<NetworkConfiguration::LayerP> &layers) const
{
  std::vector<VariableData> data;
  for (auto it = layers.begin(); it != layers.end(); it++)
  {
    if (!(*it)->hasParams())
      continue;
    auto layerPtr = std::dynamic_pointer_cast<NetworkConfiguration::ParameterizedLayer>(*it);
    auto values = layerPtr->getParamShapes();
    for (auto fwdit = values.begin(); fwdit != values.end(); fwdit++)
      data.push_back(VariableData((*fwdit).first, (*fwdit).second, (*fwdit).second.numberOfElements()));
  }
  return data;
}

size_t ModelEvalOp::totalNumberOfParameters() const
{
  return std::accumulate(m_VariableData.begin(), m_VariableData.end(), 0, [](size_t sum, const VariableData & val) { return sum + val.m_NumberOfElements;});
}


bool ModelEvalOp::initialize(StateP state)
{
  m_ECFState = state;
  try
  {
    // load parameterization and configuration data
    ECF_LOG(state, 3, "Loading network configuration...");
    std::string configFilePath = *(static_cast<std::string*> (state->getRegistry()->getEntry("configFilePath").get()));
    m_SaveModel = *(static_cast<int*> (state->getRegistry()->getEntry("saveModel").get()));
    m_ModelExportPath = *(static_cast<std::string*> (state->getRegistry()->getEntry("modelSavePath").get()));
    ConfigParser configParser(configFilePath);
    std::vector<std::pair<std::string, std::vector<std::vector<int>>>> layerConfiguration = configParser.LayerConfiguration();
    const std::vector<std::string> & datasetInputFiles = configParser.InputFiles();
    const std::vector<std::string> & datasetLabelFiles = configParser.LabelFiles();
    std::string datasetLoaderType = configParser.DatasetLoaderType();
    std::string lossFunctionName = configParser.LossFunctionName();
    unsigned int batchSize = configParser.BatchSize();
    // set input and output shapes
    NetworkConfiguration::Shape inputShape({batchSize});
    inputShape.insert(inputShape.end(), configParser.InputShape().begin(), configParser.InputShape().end());
    NetworkConfiguration::Shape outputShape({ batchSize });
    outputShape.insert(outputShape.end(), configParser.OutputShape().begin(), configParser.OutputShape().end());

    // load dataset
    ECF_LOG(state, 3, "Loading dataset...");
    m_DatasetHandler = DatasetLoader::DatasetLoaderFactory::instance().createObject(datasetLoaderType, DatasetLoader::DatasetLoaderBaseParams(datasetInputFiles, datasetLabelFiles, batchSize));

    // create network and session 
    ECF_LOG(state, 3, "Creating session...");
    // TODO: refactor this, can be much better
    m_Network = createLayers(m_Scope, layerConfiguration, lossFunctionName, inputShape, outputShape);
    m_VariableData = createVariableData(m_Network);

    // create session
    Status status;
    TF_CHECK_OK(m_Scope.ToGraphDef(&m_GraphDef));
    status = m_pSession->Create(m_GraphDef);
    if (!status.ok())
    {
      ECF_LOG_ERROR(state, "Session creation failed:\n");
      ECF_LOG_ERROR(state, status.ToString());
    }
    ECF_LOG(state, 4, "Graph definition data:");
    ECF_LOG(state, 4, m_GraphDef.DebugString());
    // override size for FloatingPoint genotype
    size_t numParameters = totalNumberOfParameters();
    state->getRegistry()->modifyEntry("FloatingPoint.dimension", (voidP) new uint(numParameters));
    // reinitialize population with updated size
    state->getPopulation()->initialize(state);

    // shuffle the dataset
    m_DatasetHandler->shuffleDataset();
    return status.ok();
  }
  catch (std::exception& e)
  {
    ECF_LOG_ERROR(state, "Errors:");
    ECF_LOG_ERROR(state, e.what());
    return false;
  }
}

std::vector<std::pair<string, tensorflow::Tensor>> ModelEvalOp::createTensorsFromGenotype(const IndividualP individual) const
{
  // for FloatingPoint genotype
  FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) individual->getGenotype().get();
  // create tensors and fill them with values
  std::vector<std::pair<string, tensorflow::Tensor>> inputs;
  inputs.reserve(m_VariableData.size());
  auto currentIterator = gen->realValue.begin();
  for (auto it = m_VariableData.begin(); it != m_VariableData.end(); it++)
  {
    Tensor tensor(DT_FLOAT, (*it).m_Shape);
    Common::setTensor<float>(tensor, currentIterator, currentIterator + (*it).m_NumberOfElements);
    currentIterator += (*it).m_NumberOfElements;
    inputs.push_back(std::make_pair((*it).m_VariableName, tensor));
  }
  return inputs;
}

FitnessP ModelEvalOp::evaluate(IndividualP individual)
{
  // set new inputs and outputs if generation has changed
  setBatch(m_ECFState->getGenerationNo());
  // set placeholders
  FitnessP fitness (new FitnessMin);
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = createTensorsFromGenotype(individual);
  inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, m_CurrentInputs));
  inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, m_CurrentOutputs));
  // run session and fetch loss
  std::vector<tensorflow::Tensor> outputs;
  Status status = m_pSession->Run(inputs, { LOSS_OUTPUT_NAME }, {}, &outputs);
  auto loss = outputs[0].scalar<float>();
  fitness->setValue(loss());
  return fitness;

}