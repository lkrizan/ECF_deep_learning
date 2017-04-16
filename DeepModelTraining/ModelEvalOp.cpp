#include "ModelEvalOp.h"
#include "ConfigParser.h"

#define INPUTS_PLACEHOLDER_NAME "inputs"
#define OUTPUTS_PLACEHOLDER_NAME "outputs"
#define LOSS_OUTPUT_NAME "loss"


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
	m_Session->Close();
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

template<class T, class InputIterator>
void ModelEvalOp::setTensor(Tensor &tensor, InputIterator first, InputIterator last)
{
    auto tensorMap = tensor.flat<T>();
    int currentIdx = 0;
    for (auto it = first; it != last; it++)
        tensorMap(currentIdx++) = static_cast<T>(*it);
}


std::vector<NetworkConfiguration::LayerP> ModelEvalOp::createLayers(Scope &root, const std::vector<std::pair<std::string, std::vector<int>>>& networkConfiguration, const std::string lossFunctionName, const NetworkConfiguration::Shape & inputShape, const NetworkConfiguration::Shape & outputShape) const
{
	using NetworkConfiguration::Shape;
	// create placeholders for inputs and for expected outputs so network can be defined
	auto inputPlaceholder = ops::Placeholder(root.WithOpName(INPUTS_PLACEHOLDER_NAME), DT_FLOAT);
	auto outputPlaceholder = ops::Placeholder(root.WithOpName(OUTPUTS_PLACEHOLDER_NAME), DT_FLOAT);
	// create layers from value pairs - layer type name and shape in form of vector of ints
	std::vector<NetworkConfiguration::LayerP> layers;
	for (auto iter = networkConfiguration.begin(); iter != networkConfiguration.end(); iter++)
	{
		Shape paramShape(iter->second.begin(), iter->second.end());
		NetworkConfiguration::LayerP layer;
		if (iter == networkConfiguration.begin())
			layer = NetworkConfiguration::LayerFactory::createLayer(iter->first, root, inputPlaceholder, inputShape, paramShape);
		else
			layer = NetworkConfiguration::LayerFactory::createLayer(iter->first, root, layers.back()->forward(), layers.back()->outputShape(), paramShape);
		layers.push_back(layer);
	}
	// add loss function to graph
	auto lossFunction = NetworkConfiguration::LossFunctionFactory::createLossFunction(lossFunctionName, root, layers.back()->forward(), layers.back()->outputShape(), outputPlaceholder, outputShape, LOSS_OUTPUT_NAME);
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
		std::vector<std::pair<std::string, std::vector<int>>> layerConfiguration = configParser.LayerConfiguration();
		int numInputs = configParser.NumInputs();
		int numOutputs = configParser.NumOutputs();
		std::string datasetPath = configParser.DatasetPath();
		std::string lossFunctionName = configParser.LossFunctionName();
		// TODO: refactor this so that inputs and output shape do not have to be matrices (they can be tensors)
		int inputShape_[] = { 0, numInputs };
		int outputShape_[] = { 0, numOutputs };
		NetworkConfiguration::Shape inputShape(begin(inputShape_), end(inputShape_));
		NetworkConfiguration::Shape outputShape(begin(outputShape_), end(outputShape_));

		// load dataset
		ECF_LOG(state, 3, "Loading dataset...");
		m_DatasetHandler = DatasetLoader::IDatasetLoaderP(new DatasetLoader::NumericDatasetLoader(datasetPath));

		// create network and 
		ECF_LOG(state, 3, "Creating session...");
		std::vector<NetworkConfiguration::LayerP> layers = createLayers(m_Scope, layerConfiguration, lossFunctionName, inputShape, outputShape);
		// layers are only used for helping in creating graph definition - layers themselves are not used anywhere else later
		// instead of layers, create instances of VariableData class which carry only required information - symbolic parameter names, their shapes and number of elements
		m_VariableData = createVariableData(layers);

		// create session
		Status status;
		SessionOptions options;
		NewSession(options, &m_Session);
		TF_CHECK_OK(m_Scope.ToGraphDef(&m_GraphDef));
		status = m_Session->Create(m_GraphDef);
		ECF_LOG(state, 5, "Graph definition data:");
		ECF_LOG(state, 5, m_GraphDef.DebugString());

		// override size for FloatingPoint genotype
		size_t numParameters = totalNumberOfParameters();
		state->getRegistry()->modifyEntry("FloatingPoint.dimension", (voidP) new uint(numParameters));
		// reinitialize population with updated size
		state->getPopulation()->initialize(state);
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
		setTensor<float>(tensor, currentIterator, currentIterator + (*it).m_NumberOfElements);
		currentIterator += (*it).m_NumberOfElements;
		inputs.push_back(std::make_pair((*it).m_VariableName, tensor));
	}
	return inputs;
}

FitnessP ModelEvalOp::evaluate(IndividualP individual)
{
    FitnessP fitness (new FitnessMin);
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = createTensorsFromGenotype(individual);
	unsigned int batchCounter = 0;
	float accumulatedBatchLoss = 0;
	Tensor datasetInputs, datasetOutputs;
	inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, datasetInputs));
	std::pair<std::string, tensorflow::Tensor> *inputsPair = &inputs.back();
	inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, datasetOutputs));
	std::pair<std::string, tensorflow::Tensor> *outputsPair = &inputs.back();
	m_DatasetHandler->resetBatchIterator();
	while (m_DatasetHandler->nextBatch(datasetInputs, datasetOutputs))
	{
		// set placeholders to batch inputs and expected outputs
		inputsPair->second = datasetInputs;
		outputsPair->second = datasetOutputs;
		// run session and fetch loss
		std::vector<tensorflow::Tensor> outputs;
		Status status = m_Session->Run(inputs, { LOSS_OUTPUT_NAME }, {}, &outputs);
		auto outputRes = outputs[0].scalar<float>();
		accumulatedBatchLoss += outputRes();
		++batchCounter;
	}
	fitness->setValue(accumulatedBatchLoss / batchCounter);
    return fitness;

}