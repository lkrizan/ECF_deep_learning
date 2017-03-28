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
        tensorMap(currentIdx++) = (T) *it;
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
	// TODO: only for testing purposes, will be replaced with reading from file and will not leak
	std::array<int, 2> inputshp_ = { 0, N_INPUTS };
	std::array<int, 2> outputshp_ = { 0, N_OUTPUTS };
	std::array<int, 2> w1shp_ = { FIRST_LAYER, N_INPUTS };
	std::array<int, 2> w2shp_ = { N_OUTPUTS, FIRST_LAYER };
	FullyConnectedLayer* first_ = new FullyConnectedLayer(x, root, Shape(inputshp_.begin(), inputshp_.end()), Shape(w1shp_.begin(), w1shp_.end()));
	SigmoidActivation* first_activation_ = new SigmoidActivation(first_->forward(), root, first_->outputShape());
	FullyConnectedLayer* second_ = new FullyConnectedLayer(first_activation_->forward(), root, first_activation_->outputShape(), Shape(w2shp_.begin(), w2shp_.end()));
	MeanSquaredLoss* loss_ = new MeanSquaredLoss(second_->forward(), y, root, second_->outputShape(), Shape(outputshp_.begin(), outputshp_.end()));
	// create session
	GraphDef def;
	TF_CHECK_OK(root.ToGraphDef(&def));
	return def;
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
    FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) individual->getGenotype().get();

    // not very nice, but it will be changed once parameters won't be hardcoded
    // create tensors
    TensorShape w1Shape({FIRST_LAYER, N_INPUTS});
    Tensor w1(DT_FLOAT, w1Shape);
    TensorShape b1Shape({FIRST_LAYER});
    Tensor b1(DT_FLOAT, b1Shape);
    TensorShape w2Shape({N_OUTPUTS, FIRST_LAYER});
    Tensor w2(DT_FLOAT, w2Shape);
    TensorShape b2Shape({N_OUTPUTS});
    Tensor b2(DT_FLOAT, b2Shape);

    // set tensor values
    auto currentIterator = gen->realValue.begin() + FIRST_LAYER * N_INPUTS;
    setTensor<float>(w1, gen->realValue.begin(), currentIterator);
    setTensor<float>(b1, currentIterator, currentIterator + FIRST_LAYER);
    currentIterator += FIRST_LAYER;
    setTensor<float>(w2, currentIterator, currentIterator + N_OUTPUTS * FIRST_LAYER);
    currentIterator += N_OUTPUTS * FIRST_LAYER;
    setTensor<float>(b2, currentIterator, gen->realValue.end());

    // setting tensor values can also be performed via std::copy
    // (but it does not cover for type difference, as far as I know)
    // std::copy(gen->realValue.begin(), gen->realValue.end(), w_map.data());
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "FC1_w", w1 },
            { "FC2_w", w2 },
            { "FC1_b", b1 },
            { "FC2_b", b2 },
            { INPUTS_PLACEHOLDER_NAME, *m_Inputs },
            { OUTPUTS_PLACEHOLDER_NAME, *m_Outputs }
    };

    std::vector<tensorflow::Tensor> outputs;
    /*
    std::cout << "W1: " << w1.DebugString() << std::endl;
    std::cout << "b1: " << b1.DebugString() << std::endl;
    std::cout << "W2: " << w2.DebugString() << std::endl;
    std::cout << "b2: " << b2.DebugString() << std::endl;
    std::cout << "X: " << m_Inputs->DebugString() << std::endl;
    std::cout << "Y: " << m_Outputs->DebugString() << std::endl;
    */
	Status status = m_Session->Run(inputs, { LOSS_OUTPUT_NAME }, {}, &outputs);
    // std::cout << status.ToString() << std::endl;
    // std::cout << outputs[0].DebugString() << std::endl;
    auto outputRes = outputs[0].scalar<float>();
    fitness->setValue(outputRes());
    return fitness;

}