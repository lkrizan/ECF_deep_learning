#ifndef MODELEVALOP_H_
#define MODELEVALOP_H_
#include <ECF/ECF.h>
#include <Layers/Layers_all.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "DatasetLoader.h"

using namespace tensorflow;

class ModelEvalOp: public EvaluateOp
{
public:
	~ModelEvalOp();
    FitnessP evaluate(IndividualP individual);
    void registerParameters(StateP);
    bool initialize(StateP);
    template <class T, class InputIterator>
    static void setTensor(Tensor &tensor, InputIterator first, InputIterator last);

private:
    Session *m_Session;
    std::shared_ptr<Tensor> m_Inputs;
    std::shared_ptr<Tensor> m_Outputs;
	GraphDef createGraphDef();
};

typedef boost::shared_ptr<ModelEvalOp> ModelEvalOpP;
#endif