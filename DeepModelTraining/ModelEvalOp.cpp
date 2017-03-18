#include <ECF/ECF.h>
#include "ModelEvalOp.h"

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
	// TODO: will be parameterized, for now everything is hardcoded for testing purposes
	Scope root = Scope::NewRootScope();
	using namespace tensorflow::ops;
	// placeholders for inputs and outputs
	auto x = Placeholder(root.WithOpName("x"), DT_FLOAT);
	auto y = Placeholder(root.WithOpName("y"), DT_FLOAT);
	// placeholders for network parameters
	auto w1 = Placeholder(root.WithOpName("w1"), DT_FLOAT);
	auto b1 = Placeholder(root.WithOpName("b1"), DT_FLOAT);
	auto w2 = Placeholder(root.WithOpName("w2"), DT_FLOAT);
	auto b2 = Placeholder(root.WithOpName("b2"), DT_FLOAT);
	// network definition
	auto firstLayer = MatMul(root, x, w1, ops::MatMul::TransposeB(true));
	auto firstLayerNet = Add(root, firstLayer, b1);
	auto firstLayerOut = Sigmoid(root, firstLayerNet);
	auto secondLayer = MatMul(root, firstLayerOut, w2, ops::MatMul::TransposeB(true));
	auto secondLayerOut = Add(root, secondLayer, b2);
	auto diff = Subtract(root, secondLayerOut, y);
	auto stDeviation = Square(root, diff);
	auto meanLoss = Mean(root.WithOpName("loss"), stDeviation, 0);
	// create session
	GraphDef def;
	TF_CHECK_OK(root.ToGraphDef(&def));
	return def;
}


bool ModelEvalOp::initialize(StateP state)
{
	// create session
	SessionOptions options;
	NewSession(options, &m_Session);
	GraphDef graphDef = createGraphDef();
	Status status = m_Session->Create(graphDef);
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
	return status.ok();
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
            { "w1", w1 },
            { "w2", w2 },
            { "b1", b1 },
            { "b2", b2 },
            { "x", *m_Inputs },
            { "y", *m_Outputs }
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
	Status status = m_Session->Run(inputs, { "loss" }, {}, &outputs);
    // std::cout << status.ToString() << std::endl;
    // std::cout << outputs[0].DebugString() << std::endl;
    auto outputRes = outputs[0].scalar<float>();
    fitness->setValue(outputRes());
    return fitness;

}