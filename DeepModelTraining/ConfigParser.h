#include <vector>
#include <fstream>
#include <boost/tokenizer.hpp>


class ConfigParser
{
private:
	std::vector<std::pair<std::string, std::vector<int>>> m_LayerConfiguration;
	int m_NumInputs;
	int m_NumOutputs;
	std::string m_DatasetPath;

	enum State {eStart, eGeneral, eLayers};
	State m_State = eStart;

	void parseLine(const std::string line);
	bool isHeaderLine(const std::string line) const { return (line.size() > 2 && line.front() == '[' && line.back() == ']'); }

public:
	ConfigParser(const std::string pathToFile);
	std::vector<std::pair<std::string, std::vector<int>>> LayerConfiguration() { return m_LayerConfiguration; }
	int NumInputs() { return m_NumInputs; }
	int NumOutputs() { return m_NumOutputs; }
	std::string DatasetPath() { return m_DatasetPath; }
};
