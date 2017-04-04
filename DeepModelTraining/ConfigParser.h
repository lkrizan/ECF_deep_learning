#include <vector>
#include <fstream>
#include <boost/tokenizer.hpp>


class ConfigParser
{

// TODO: parser should check if all required parameters are included

private:
	std::vector<std::pair<std::string, std::vector<int>>> m_LayerConfiguration;
	int m_NumInputs;
	int m_NumOutputs;
	std::string m_DatasetPath;
	std::string m_LossFunctionName;

	// used for checking if all required parameters are configured
	bool inputsConfigured = false;
	bool outputsConfigured = false;
	bool datasetPathConfigured = false;
	bool layerConfigurationConfigured = false;
	bool lossFunctionConfigured = false;

	enum State {eStart, eGeneral, eLayers, eLoss, eLossFinished};
	State m_State = eStart;

	void parseLine(const std::string line);
	bool isHeaderLine(const std::string line) const { return (line.size() > 2 && line.front() == '[' && line.back() == ']'); }

public:
	ConfigParser(const std::string pathToFile);
	std::vector<std::pair<std::string, std::vector<int>>> LayerConfiguration() { return m_LayerConfiguration; }
	int NumInputs() { return m_NumInputs; }
	int NumOutputs() { return m_NumOutputs; }
	std::string DatasetPath() { return m_DatasetPath; }
	std::string LossFunctionName() { return m_LossFunctionName; }
};
