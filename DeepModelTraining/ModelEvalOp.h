#ifndef MODELEVALOP_H_
#define MODELEVALOP_H_
#include <ECF/ECF.h>
#include <NetworkConfiguration/Layer.h>
#include <NetworkConfiguration/LossFunction.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include "ModelExporter.h"
#include <DatasetLoader/NumericDatasetLoader.h>

using namespace tensorflow;

class ModelEvalOp: public EvaluateOp
{
public:
  ModelEvalOp() : m_pSession(NewSession({})) {}
  ~ModelEvalOp();
  FitnessP evaluate(IndividualP individual);
  void registerParameters(StateP);
  bool initialize(StateP);
  template <class T, class InputIterator>
  static void setTensor(Tensor &tensor, InputIterator first, InputIterator last);

private:

  struct VariableData
  {
    NetworkConfiguration::Shape m_BasicShape;
    std::string m_VariableName;
    TensorShape m_Shape;
    int m_NumberOfElements;
    VariableData(std::string variableName, NetworkConfiguration::Shape shape, int numberOfElements) : m_VariableName(variableName), m_BasicShape(shape), m_Shape(shape.asTensorShape()), m_NumberOfElements(numberOfElements) {}
  };
  StateP m_ECFState = nullptr;

  std::unique_ptr<Session> m_pSession;

  GraphDef m_GraphDef;
  Scope m_Scope = Scope::NewRootScope();
  std::vector<VariableData> m_VariableData;

  bool m_SaveModel = false;
  std::string m_ModelExportPath;

  DatasetLoader::IDatasetLoaderP m_DatasetHandler;

  // save graph definition and tensor values to disk
  void saveDefinitionToFile() const;
  // helper function for filling tensors with values from genotype
  std::vector<std::pair<string, tensorflow::Tensor>> createTensorsFromGenotype(const IndividualP individual) const;
  // helper function for creating graph definition
  std::vector<NetworkConfiguration::LayerP> createLayers(Scope &root, const std::vector<std::pair<std::string, std::vector<std::vector<int>>>> & networkConfiguration, const std::string lossFunctionName, const NetworkConfiguration::Shape & inputShape, const NetworkConfiguration::Shape & outputShape) const;
  // helper function for creating variable data
  std::vector<VariableData> createVariableData(const std::vector<NetworkConfiguration::LayerP> &layers) const;
  // helper function which calculates total number of parameters from network configuration - used for overriding size of FloatingPoint genotype
  size_t totalNumberOfParameters() const;

  // used for tracking dataset batches (every generation - new batch) 
  int m_CurrentGeneration = -1;
  Tensor m_CurrentInputs;
  Tensor m_CurrentOutputs;
};

typedef boost::shared_ptr<ModelEvalOp> ModelEvalOpP;
#endif