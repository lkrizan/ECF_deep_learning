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
	struct VariableData 
	{
		std::string m_VariableName;
		TensorShape m_Shape;
		int m_NumberOfElements;
		VariableData(std::string variableName, TensorShape shape, int numberOfElements) : m_VariableName(variableName), m_Shape(shape), m_NumberOfElements(numberOfElements) {}
	};
    Session *m_Session;
	std::vector<Layers::LayerP> m_Layers;
	std::vector<VariableData> m_VariablesData;
    std::shared_ptr<Tensor> m_Inputs;
    std::shared_ptr<Tensor> m_Outputs;
	GraphDef createGraphDef();
	void createVariableData();
};

typedef boost::shared_ptr<ModelEvalOp> ModelEvalOpP;
#endif