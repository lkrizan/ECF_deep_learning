#include "NumericDatasetLoader.h"

namespace DatasetLoader {

void NumericDatasetLoader::parseLine(const std::string & line, std::vector<float>& values) const
{
  values.clear();
  typedef boost::tokenizer<boost::char_separator<char>> tok_t;
  boost::char_separator<char> sep(" ");
  tok_t tok(line, sep);
  values.resize(std::distance(tok.begin(), tok.end()));
  std::transform(tok.begin(), tok.end(), values.begin(), [](const std::string val) { return std::stof(val); });
}


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
    parseLine(line, values);
    if (values.size() != 2)
      throw std::logic_error("First line of dataset file should contain number of inputs and outputs.");
    m_LearningExampleShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[0]) });
    m_LabelShape = NetworkConfiguration::Shape({ static_cast<unsigned int>(values[1]) });
  }
  // read all other values
  while (getline(fileP, line))
  {
    parseLine(line, values);
    addLearningExample(values.begin(), values.begin() + m_LearningExampleShape.front());
    addLabel(values.begin() + m_LearningExampleShape.front(), values.end());
  }
  fileP.close();

  // set iterators so dataset can be used
  resetBatchIterator();
}

}	// namespace DatasetLoader
