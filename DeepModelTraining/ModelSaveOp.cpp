#include "ModelSaveOp.h"

#define INFO_FILE_NAME "info.dat"

double ModelSaveOp::getAverageTrainingError(IndividualP individual)
{
  DatasetLoader::IDatasetLoaderP datasetHandler = m_EvaluationOp->m_DatasetHandler;
  // reset datasetHandler's batch iterators and stop evaluation op from changing batches
  datasetHandler->resetBatchIterator();
  double avgError = 0;
  int counter = 0;
  while (datasetHandler->nextBatch(m_EvaluationOp->m_CurrentInputs, m_EvaluationOp->m_CurrentOutputs))
  {
    avgError += m_EvaluationOp->evaluate(individual)->getValue();
    ++counter;
  }
  return avgError / counter;
}

void ModelSaveOp::registerParameters(StateP state)
{
  state->getRegistry()->registerEntry("saveModel", (voidP)(new int(0)), ECF::INT);
  state->getRegistry()->registerEntry("modelSavePath", (voidP)(new std::string), ECF::STRING);
}

bool ModelSaveOp::initialize(StateP state)
{
  if (!state->getRegistry()->isModified("term.maxgen"))
    return false;
  m_SelBestOp->initialize(state);
  m_TerminatingGeneration = *static_cast<uint*>(state->getRegistry()->getEntry("term.maxgen").get());
  m_EvaluationOp = dynamic_pointer_cast<ModelEvalOp>(state->getEvalOp());
  m_SaveModel = *(static_cast<int*> (state->getRegistry()->getEntry("saveModel").get()));
  m_ModelExportPath = *(static_cast<std::string*> (state->getRegistry()->getEntry("modelSavePath").get()));
  // if evaluation operator is not ModelEvalOp
  if (nullptr == m_EvaluationOp)
  {
    ECF_LOG_ERROR(state, "ModelSaveOp should only be used with ModelEvalOp evaluation operator.");
    return false;
  }
  return true;
}

bool ModelSaveOp::operate(StateP state)
{
  if (m_SaveModel)
  {
    if (m_TerminatingGeneration == state->getGenerationNo())
    {
      // get the best individual from current population
      DemeP deme = state->getPopulation()->at(0);
      IndividualP best = m_SelBestOp->select(*deme);
      double avgError = getAverageTrainingError(best);
      // TODO: create directory and set path
      /*
      boost::filesystem::path dir(m_ModelExportPath.c_str());
      if (!boost::filesystem::create_directory(dir))
      {
        ECF_LOG(state, 2, "Failed to create model export directory. Saving to root directory.");
        m_ModelExportPath = "";
      }
      else
      {
        m_ModelExportPath += "\\";
      }
      */
      try
      {
        ModelExporter exporter(state, m_ModelExportPath);
        // save graph definition and variables
        exporter.exportGraph(m_EvaluationOp->m_GraphDef);
        FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) best->getGenotype().get();
        auto currentIterator = gen->realValue.begin();
        auto endIterator = gen->realValue.end();
        for_each(m_EvaluationOp->m_VariableData.begin(), m_EvaluationOp->m_VariableData.end(), [&exporter, &currentIterator](const ModelEvalOp::VariableData & data) {exporter.exportVariableValues(data.m_VariableName, data.m_BasicShape, currentIterator, currentIterator + data.m_NumberOfElements); currentIterator += data.m_NumberOfElements;});
        // save additional data (only avg training error for now)
        m_OutputStream.open(exporter.m_FolderPath + INFO_FILE_NAME);
        if (m_OutputStream.is_open())
        {
          m_OutputStream << "Avg train error" << std::endl;
          m_OutputStream << avgError << std::endl;
          m_OutputStream.close();
        }
      }
      catch (std::exception &e)
      {
        std::string errMsg = "Failed to save trained model: " + std::string(e.what());
        ECF_LOG_ERROR(state, errMsg);
      }
    }
  }
}

