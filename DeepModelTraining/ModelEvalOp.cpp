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
	m_Session->Close();
}

template<class T, class InputIterator>
void ModelEvalOp::setTensor(Tensor &tensor, InputIterator first, InputIterator last)
{
    auto tensorMap = tensor.flat<T>();
    int currentIdx = 0;
    for (auto it = first; it != last; it++)
        tensorMap(currentIdx++) = static_cast<T>(*it);
}

GraphDef ModelEvalOp::createGraphDef(const std::vector<std::pair<std::string, std::vector<int>>> & networkConfiguration, const std::string lossFunctionName, const NetworkConfiguration::Shape & inputShape, const NetworkConfiguration::Shape & outputShape)
{
	using NetworkConfiguration::Shape;
	Scope root = Scope::NewRootScope();
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
	// layers are only used for helping in creating graph definition - layers themselves are not used anywhere else later
	// instead of layers, create instances of VariableData class which carry only required information - symbolic parameter names, their shapes and number of elements
	createVariableData(layers);
	// create session
	GraphDef def;
	TF_CHECK_OK(root.ToGraphDef(&def));
	return def;
}

void ModelEvalOp::createVariableData(const std::vector<NetworkConfiguration::LayerP> &layers)
{
	for (auto it = layers.begin(); it != layers.end(); it++)
	{
		if (!(*it)->hasParams())
			continue;
		auto layerPtr = std::dynamic_pointer_cast<NetworkConfiguration::ParameterizedLayer>(*it);
		auto values = layerPtr->getParamShapes();
		for (auto fwdit = values.begin(); fwdit != values.end(); fwdit++)
			m_VariablesData.push_back(VariableData((*fwdit).first, (*fwdit).second.asTensorShape(), (*fwdit).second.numberOfElements()));
	}
}


bool ModelEvalOp::initialize(StateP state)
{
	try
	{
		// load parameterization data
		std::string configFilePath = *(static_cast<std::string*> (state->getRegistry()->getEntry("configFilePath").get()));
		ConfigParser configParser(configFilePath);
		std::vector<std::pair<std::string, std::vector<int>>> layerConfiguration = configParser.LayerConfiguration();
		int numInputs = configParser.NumInputs();
		int numOutputs = configParser.NumOutputs();
		std::string datasetPath = configParser.DatasetPath();
		std::string lossFunctionName = configParser.LossFunctionName();
		// load training data
		DatasetLoader<float> datasetParser(datasetPath, numInputs, numOutputs);
		std::vector<float> inputs = datasetParser.getInputs();
		std::vector<float> outputs = datasetParser.getOutputs();
		// TODO: refactor this so that inputs and output shape do not have to be matrices (they can be tensors)
		int inputShape_[] = { inputs.size() / numInputs, numInputs };
		int outputShape_[] = { outputs.size() / numOutputs, numOutputs };
		NetworkConfiguration::Shape inputShape(begin(inputShape_), end(inputShape_));
		NetworkConfiguration::Shape outputShape(begin(outputShape_), end(outputShape_));
		// create session
		Status status;
		SessionOptions options;
		NewSession(options, &m_Session);
		GraphDef graphDef = createGraphDef(layerConfiguration, lossFunctionName, inputShape, outputShape);
		status = m_Session->Create(graphDef);
		// create tensors for inputs and outputs and fill them with values from dataset
		m_Inputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, inputShape.asTensorShape()));
		setTensor<float>(*m_Inputs, inputs.begin(), inputs.end());
		m_Outputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, outputShape.asTensorShape()));
		setTensor<float>(*m_Outputs, outputs.begin(), outputs.end());
		return status.ok();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
}

FitnessP ModelEvalOp::evaluate(IndividualP individual)
{
    FitnessP fitness (new FitnessMin);
	// for FloatingPoint genotype
    FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) individual->getGenotype().get();
	// create tensors and fill them with values
	std::vector<std::pair<string, tensorflow::Tensor>> inputs;
	inputs.reserve(m_VariablesData.size());
	auto currentIterator = gen->realValue.begin();
	for (auto it = m_VariablesData.begin(); it != m_VariablesData.end(); it++)
	{
		Tensor tensor(DT_FLOAT, (*it).m_Shape);
		setTensor<float>(tensor, currentIterator, currentIterator + (*it).m_NumberOfElements);
		currentIterator += (*it).m_NumberOfElements;
		inputs.push_back(std::make_pair((*it).m_VariableName, tensor));
	}
	inputs.push_back(std::make_pair(INPUTS_PLACEHOLDER_NAME, *m_Inputs));
	inputs.push_back(std::make_pair(OUTPUTS_PLACEHOLDER_NAME, *m_Outputs));
    std::vector<tensorflow::Tensor> outputs;
	Status status = m_Session->Run(inputs, { LOSS_OUTPUT_NAME }, {}, &outputs);
    auto outputRes = outputs[0].scalar<float>();
    fitness->setValue(outputRes());
    return fitness;

}