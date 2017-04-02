#include "ModelEvalOp.h"
#include <array>

#define N_INPUTS 2
#define N_OUTPUTS 1
#define FIRST_LAYER 10


void ModelEvalOp::registerParameters(StateP state)
{
    // does nothing for now
    // will be updated when models will have parameters
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

GraphDef ModelEvalOp::createGraphDef()
{
	using Layers::Shape;
	// TODO: will be parameterized, for now everything is hardcoded for testing purposes
	Scope root = Scope::NewRootScope();
	using namespace tensorflow::ops;
	// placeholders for inputs and outputs
	auto x = Placeholder(root.WithOpName(INPUTS_PLACEHOLDER_NAME), DT_FLOAT);
	auto y = Placeholder(root.WithOpName(OUTPUTS_PLACEHOLDER_NAME), DT_FLOAT);
	// TODO: only for testing purposes
	std::array<int, 2> inputshp_ = { 0, N_INPUTS };
	std::array<int, 2> outputshp_ = { 0, N_OUTPUTS };
	std::array<int, 2> w1shp_ = { FIRST_LAYER, N_INPUTS };
	std::array<int, 2> w2shp_ = { N_OUTPUTS, FIRST_LAYER };
	Layers::LayerP first_ (new Layers::FullyConnectedLayer(x, root, Shape(inputshp_.begin(), inputshp_.end()), Shape(w1shp_.begin(), w1shp_.end())));
	m_Layers.push_back(first_);
	Layers::LayerP first_activation_ (new Layers::SigmoidActivation(first_->forward(), root, first_->outputShape()));
	m_Layers.push_back(first_activation_);
	Layers::LayerP second_ (new Layers::FullyConnectedLayer(first_activation_->forward(), root, first_activation_->outputShape(), Shape(w2shp_.begin(), w2shp_.end())));
	m_Layers.push_back(second_);
	Layers::LayerP loss_ (new Layers::MeanSquaredLoss(second_->forward(), y, root, second_->outputShape(), Shape(outputshp_.begin(), outputshp_.end())));
	m_Layers.push_back(loss_);
	// create session
	GraphDef def;
	TF_CHECK_OK(root.ToGraphDef(&def));
	return def;
}

void ModelEvalOp::createVariableData()
{
	for (auto it = m_Layers.begin(); it != m_Layers.end(); it++)
	{
		if (!(*it)->hasParams())
			continue;
		auto layerPtr = std::dynamic_pointer_cast<Layers::ParameterizedLayer>(*it);
		auto values = layerPtr->getParamShapes();
		for (auto fwdit = values.begin(); fwdit != values.end(); fwdit++)
			m_VariablesData.push_back(VariableData((*fwdit).first, (*fwdit).second.asTensorShape(), (*fwdit).second.numberOfElements()));
	}
}


bool ModelEvalOp::initialize(StateP state)
{
    // load training data
    DatasetLoader<float> parser("./dataset/dataset.txt", N_INPUTS, N_OUTPUTS);
    std::vector<float> inputs = parser.getInputs();
    std::vector<float> outputs = parser.getOutputs();
    // create tensors and set their values
    TensorShape inputShape({(int64)inputs.size() / N_INPUTS, N_INPUTS});
    m_Inputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, inputShape));
    setTensor<float>(*m_Inputs, inputs.begin(), inputs.end());
    TensorShape outputShape({(int64) outputs.size(), N_OUTPUTS});
    m_Outputs = std::make_shared<Tensor>(Tensor(DT_FLOAT, outputShape));
    setTensor<float>(*m_Outputs, outputs.begin(), outputs.end());
	// create session
	try
	{
		SessionOptions options;
		NewSession(options, &m_Session);
		GraphDef graphDef = createGraphDef();
		Status status = m_Session->Create(graphDef);
		createVariableData();
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