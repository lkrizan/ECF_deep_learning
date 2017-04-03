#ifndef MODELEVALOP_H_
#define MODELEVALOP_H_
#include <ECF/ECF.h>
#include <NetworkConfiguration/NetworkConfiguration.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "DatasetLoader.h"

using namespace tensorflow;

// TODO: check if dimensions of FloatingPoint genotype fit dimensions from configuration (or do something even more clever)
// TODO: use ECF logging instead of direct output to stdout

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
	std::vector<VariableData> m_VariablesData;
    std::shared_ptr<Tensor> m_Inputs;
    std::shared_ptr<Tensor> m_Outputs;
	// helper function for creating graph definition
	GraphDef createGraphDef(const std::vector<std::pair<std::string, std::vector<int>>> & networkConfiguration, const std::string lossFunctionName, const NetworkConfiguration::Shape & inputShape, const NetworkConfiguration::Shape & outputShape);
	// helper function for creating variable data
	void createVariableData(const std::vector<NetworkConfiguration::LayerP> &layers);
};

typedef boost::shared_ptr<ModelEvalOp> ModelEvalOpP;
#endif