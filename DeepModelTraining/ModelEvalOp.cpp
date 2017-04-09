#include "ModelEvalOp.h"
#include "ConfigParser.h"

#define INPUTS_PLACEHOLDER_NAME "inputs"
#define OUTPUTS_PLACEHOLDER_NAME "outputs"
#define LOSS_OUTPUT_NAME "loss"


void ModelEvalOp::registerParameters(StateP state)
{
	state->getRegistry()->registerEntry("configFilePath", (voidP)(new std::string), ECF::STRING);
}

ModelEvalOp::~ModelEvalOp()
{
	if (m_SaveModel)
	{
		try
		{
			saveDefinitionToFile(m_ModelExportPath);
		}
		catch (std::exception &e)
		{
			std::string errMsg = "Failed to save trained model: " + std::string(e.what());
			ECF_LOG_ERROR(m_ECFState, errMsg);
		}
	}
	m_Session->Close();
}

void ModelEvalOp::saveDefinitionToFile(std::string folderPath) const
{
	// save graph definition
	tensorflow::WriteBinaryProto(tensorflow::Env::Default(), folderPath + "\\graph.pb", m_GraphDef);
	// get best individual from Hall of Fame
	IndividualP bestIndividual = m_ECFState->getHoF()->getBest().at(0);
	/*
	// adjust data types - because google likes to complicate everything (python bindings use normal types)
	std::vector<std::pair<std::string, tensorflow::Tensor>> values = createTensorsFromGenotype(bestIndividual);
	std::vector<std::string> tensorNames;
	std::vector<tensorflow::Input> tensorVals;
	tensorNames.resize(values.size());
	tensorVals.resize(values.size());
	std::transform(values.begin(), values.end(), tensorNames.begin(), [](const std::pair<std::string, tensorflow::Tensor> &val) { return val.first; });
	std::transform(values.begin(), values.end(), tensorVals.begin(), [](const std::pair<std::string, tensorflow::Tensor> &val) { return Input(val.second); });
	Tensor names(DataType::DT_STRING, TensorShape({ static_cast<int64>(values.size()) }));
	setTensor<std::string>(names, tensorNames.begin(), tensorNames.end());
	tensorflow::ops::Save(m_Scope, folderPath + "\\model.ckpt", names, tensorflow::InputList(gtl::ArraySlice<Input>(tensorVals)));
	*/
	// this part here does not work with error 'tensorflow::Input &tensorflow::Input::operator =(const tensorflow::Input &)': attempting to reference a deleted function
	// TODO: make your own format and function for saving tensors (with blackjack and hookers), and python implementation
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
			data.push_back(VariableData((*fwdit).first, (*fwdit).second.asTensorShape(), (*fwdit).second.numberOfElements()));
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
		ECF_LOG(state, 3, "Loading network configuration...");
		// load parameterization data
		std::string configFilePath = *(static_cast<std::string*> (state->getRegistry()->getEntry("configFilePath").get()));
		ConfigParser configParser(configFilePath);
		std::vector<std::pair<std::string, std::vector<int>>> layerConfiguration = configParser.LayerConfiguration();
		int numInputs = configParser.NumInputs();
		int numOutputs = configParser.NumOutputs();
		std::string datasetPath = configParser.DatasetPath();
		std::string lossFunctionName = configParser.LossFunctionName();
		ECF_LOG(state, 3, "Loading dataset...");
		// load training data
		DatasetLoader<float> datasetParser(datasetPath, numInputs, numOutputs);
		std::vector<float> inputs = datasetParser.getInputs();
		std::vector<float> outputs = datasetParser.getOutputs();
		// TODO: refactor this so that inputs and output shape do not have to be matrices (they can be tensors)
		int inputShape_[] = { inputs.size() / numInputs, numInputs };
		int outputShape_[] = { outputs.size() / numOutputs, numOutputs };
		NetworkConfiguration::Shape inputShape(begin(inputShape_), end(inputShape_));
		NetworkConfiguration::Shape outputShape(begin(outputShape_), end(outputShape_));
		ECF_LOG(state, 3, "Creating session...");
		// create network
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
		
		// create tensors for inputs and outputs and fill them with values from dataset
		m_Inputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, inputShape.asTensorShape()));
		setTensor<float>(*m_Inputs, inputs.begin(), inputs.end());
		m_Outputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, outputShape.asTensorShape()));
		setTensor<float>(*m_Outputs, outputs.begin(), outputs.end());
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
	inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, *m_Inputs));
	inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, *m_Outputs));
    std::vector<tensorflow::Tensor> outputs;
	Status status = m_Session->Run(inputs, { LOSS_OUTPUT_NAME }, {}, &outputs);
    auto outputRes = outputs[0].scalar<float>();
    fitness->setValue(outputRes());
    return fitness;

}