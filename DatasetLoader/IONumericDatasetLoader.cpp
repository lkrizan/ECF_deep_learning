#include "IONumericDatasetLoader.h"

namespace DatasetLoader {

IONumericDatasetLoader::IONumericDatasetLoader(const std::string inputsPath, const std::string labelsPath, const unsigned int batchSize) : DatasetLoader(batchSize)
{
  std::string line;
  std::vector<float> values;
  // read input file
  std::ifstream fileP(inputsPath);
  if (!fileP.is_open())
    throw std::logic_error("Failed to open dataset file " + inputsPath);
  // read first line with input shape
  if (getline(fileP, line))
  {
    splitLine(line, values);
    if (values.size() != 1)
      throw std::logic_error("First line of dataset file should contain number of inputs and outputs.");
    m_LearningExampleShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[0]) });
  }
  while (getline(fileP, line))
  {
    splitLine(line, values);
    addLearningExample(values.begin(), values.end());
  }
  fileP.close();
  // read labels file
  fileP = std::ifstream(labelsPath);
  if (!fileP.is_open())
    throw std::logic_error("Failed to open dataset file " + labelsPath);
  // read first line with output shape
  if (getline(fileP, line))
  {
    splitLine(line, values);
    if (values.size() != 1)
      throw std::logic_error("First line of dataset file should contain number of inputs and outputs.");
    m_LabelShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[0]) });
  }
  while (getline(fileP, line))
  {
    splitLine(line, values);
    addLabel(values.begin(), values.end());
  }
  fileP.close();
  // all done
  resetBatchIterator();
}

}   // namespace DatasetLoader

// register class in factory
namespace {
  using namespace DatasetLoader;
  DatasetLoaderCreator ctor = [](DatasetLoaderBaseParams & params) {return new IONumericDatasetLoader(params);};
  bool dummy = DatasetLoaderFactory::instance().registerClass("IONumericDatasetLoader", ctor);
}
