#include <vector>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <common/Shape.h>


class ConfigParser
{

private:
  std::vector<std::pair<std::string, std::vector<NetworkConfiguration::Shape>>> m_LayerConfiguration;
  std::vector<unsigned int> m_InputShape;
  std::vector<unsigned int> m_OutputShape;
  std::vector<std::string> m_InputFiles;
  std::vector<std::string> m_LabelFiles;
  unsigned int m_BatchSize = 0;
  std::string m_DatasetLoaderType;
  std::string m_LossFunctionName;

  // used for checking if all required parameters are configured
  bool inputsConfigured = false;
  bool outputsConfigured = false;
  bool datasetPathConfigured = false;
  bool layerConfigurationConfigured = false;
  bool lossFunctionConfigured = false;
  bool datasetLoaderTypeConfigured = false;

  enum State {eStart, eDataset, eLayers, eLoss, eLossFinished};
  State m_State = eStart;

  void parseLine(const std::string line);
  bool isHeaderLine(const std::string line) const { return (line.size() > 2 && line.front() == '[' && line.back() == ']'); }

public:
  ConfigParser(const std::string pathToFile);
  std::vector<std::pair<std::string, std::vector<NetworkConfiguration::Shape>>> LayerConfiguration() { return m_LayerConfiguration; }
  const std::vector<unsigned int>& InputShape() { return m_InputShape; }
  const std::vector<unsigned int>& OutputShape() { return m_OutputShape; }
  const std::vector<std::string> & InputFiles() { return m_InputFiles; }
  const std::vector<std::string> & LabelFiles() { return m_LabelFiles; }
  std::string LossFunctionName() { return m_LossFunctionName; }
  std::string DatasetLoaderType() { return m_DatasetLoaderType; }
  unsigned int BatchSize() { return m_BatchSize; }
};
