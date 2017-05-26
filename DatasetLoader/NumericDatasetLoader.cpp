#include "NumericDatasetLoader.h"

namespace DatasetLoader {

NumericDatasetLoader::NumericDatasetLoader(const std::string datasetPath, const unsigned int batchSize) : DatasetLoader(batchSize)
{
  std::ifstream fileP(datasetPath);
  if (!fileP.is_open())
    throw std::logic_error("Failed to open dataset file " + datasetPath);

  std::string line;
  std::vector<float> values;
  // read first line with number of inputs and outputs
  if (getline(fileP, line))
  {
    splitLine(line, values);
    if (values.size() != 2)
      throw std::logic_error("First line of dataset file should contain number of inputs and outputs.");
    m_LearningExampleShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[0]) });
    m_LabelShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[1]) });
  }
  // read all other values
  while (getline(fileP, line))
  {
    splitLine(line, values);
    addLearningExample(values.begin(), values.begin() + m_LearningExampleShape.front());
    addLabel(values.begin() + m_LearningExampleShape.front(), values.end());
  }
  fileP.close();

  // set iterators so dataset can be used
  resetBatchIterator();
}

}	// namespace DatasetLoader


// register class in factory
namespace {
  using namespace DatasetLoader;
  DatasetLoaderCreator ctor = [](DatasetLoaderBaseParams & params) {return new NumericDatasetLoader(params);};
  bool dummy = DatasetLoaderFactory::instance().registerClass("NumericDatasetLoader", ctor);
}
