#include "ModelSaveOp.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

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
  // should I save or should I no? (bad puns)
  state->getRegistry()->registerEntry("saveModel", (voidP)(new int(0)), ECF::INT);
  // root directory for saved models (defaults to directory from which is called if not set, or does not exist)
  state->getRegistry()->registerEntry("modelSavePath", (voidP)(new std::string), ECF::STRING);
  // save every n generations
  state->getRegistry()->registerEntry("saveModelMilestone", (voidP)(new uint(0)), ECF::UINT);
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
  m_SaveMilestone = *static_cast<uint*>(state->getRegistry()->getEntry("saveModelMilestone").get());
  m_TerminatingFitness = *static_cast<double*>(state->getRegistry()->getEntry("term.fitnessval").get());
  // if evaluation operator is not ModelEvalOp
  if (nullptr == m_EvaluationOp)
  {
    ECF_LOG_ERROR(state, "ModelSaveOp should only be used with ModelEvalOp evaluation operator.");
    return false;
  }
  // get current date and time to create unique name for the directory
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  std::string name = oss.str();
  bool status = CreateDirectory((m_ModelExportPath + "/" + name + "/").c_str(), NULL);
  if (status)
  {
    m_ModelExportPath += "/" + name + "/";
  }
  else
  {
    // creating directory failed, create it in root
    m_ModelExportPath = "./" + name + "/";
    CreateDirectory(m_ModelExportPath.c_str(), NULL);
  }
  return true;
}

bool ModelSaveOp::operate(StateP state)
{
  if (m_SaveModel)
  {
    uint currGeneration = state->getGenerationNo();
    bool bTerminatingGeneration = m_TerminatingGeneration == currGeneration;
    bool bMilestoneGeneration = (m_SaveMilestone != 0 && currGeneration != 0 && (currGeneration % m_SaveMilestone == 0));
    bool bTerminalFitness = state->getStats()->getFitnessMin() <= m_TerminatingFitness;
    // for some reason, ECF does not terminate on terminal fitness value, so here it is forced
    if(bTerminalFitness)
      state->setTerminateCond();
    if (bTerminatingGeneration || bMilestoneGeneration || bTerminalFitness)
    {
      // get the best individual from current population
      DemeP deme = state->getPopulation()->at(0);
      IndividualP best = m_SelBestOp->select(*deme);
      
      // create new directory for milestone
      std::string directoryName = "generation_" + std::to_string(currGeneration) + "/";
      CreateDirectory((m_ModelExportPath + directoryName).c_str(), NULL);

      try
      {
        ModelExporter exporter(state, m_ModelExportPath + directoryName);
        // save graph definition and variables
        exporter.exportGraph(m_EvaluationOp->m_GraphDef);
        FloatingPoint::FloatingPoint* gen = (FloatingPoint::FloatingPoint*) best->getGenotype().get();
        auto currentIterator = gen->realValue.begin();
        auto endIterator = gen->realValue.end();
        for_each(m_EvaluationOp->m_VariableData.begin(), m_EvaluationOp->m_VariableData.end(), [&exporter, &currentIterator](const ModelEvalOp::VariableData & data) {exporter.exportVariableValues(data.m_VariableName, data.m_BasicShape, currentIterator, currentIterator + data.m_NumberOfElements); currentIterator += data.m_NumberOfElements;});
        /*
        // TODO: place this in ModelExporter
        // save additional data (only avg training error for now)
        double avgError = getAverageTrainingError(best);
        m_OutputStream.open(exporter.m_FolderPath + INFO_FILE_NAME);
        if (m_OutputStream.is_open())
        {
          m_OutputStream << "Avg train error" << std::endl;
          m_OutputStream << avgError << std::endl;
          m_OutputStream.close();
        }
        */
      }
      catch (std::exception &e)
      {
        std::string errMsg = "Failed to save trained model: " + std::string(e.what());
        ECF_LOG_ERROR(state, errMsg);
      }
    }
  }
}

