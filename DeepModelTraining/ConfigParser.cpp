#include "ConfigParser.h"

void ConfigParser::parseLine(const std::string line)
{
	typedef boost::tokenizer<boost::char_separator<char>> tok_t;
	boost::char_separator<char> sep(" ");
	tok_t tokens(line, sep);
	if (tokens.begin() != tokens.end())
	{
		auto currIterator = tokens.begin();
		if (isHeaderLine(*currIterator))
		{
			// this is a header line - determine next state
			if ((*currIterator).find("General") != std::string::npos)
				m_State = eGeneral;
			else if ((*currIterator).find("Layers") != std::string::npos)
				m_State = eLayers;
			else
				throw std::logic_error(*currIterator + " is not registered block in configuration file.\n");
			return;
		}

		switch (m_State)
		{
			case eStart:
				throw std::logic_error("Parameters cannot be declared outside of blocks in configuration file.\n");
				break;
			case eGeneral:
			{
				std::string errorMsg = "Parameters in general block require argument.\n";
				if (*currIterator == "NumInputs")
				{
					if (++currIterator != tokens.end())
						m_NumInputs = std::stoi(*currIterator);
					else
						throw std::logic_error(errorMsg);
				}
				else if (*currIterator == "NumOutputs")
				{
					if (++currIterator != tokens.end())
						m_NumOutputs = std::stoi(*currIterator);
					else
						throw std::logic_error(errorMsg);
				}
				else if (*currIterator == "DatasetPath")
				{
					if (++currIterator != tokens.end())
						m_DatasetPath = *currIterator;
					else
						throw std::logic_error(errorMsg);
				}
				else
					throw std::logic_error(*currIterator + " is not a registered parameter in General block in configuration file.\n");
				break;
			}
			case eLayers:
			{
				std::vector<int> shapeValues;
				std::string layerName = *currIterator;
				shapeValues.resize(std::distance(++currIterator, tokens.end()));
				std::transform(currIterator, tokens.end(), shapeValues.begin(), [](const std::string& val) { return std::stoi(val); });
				m_LayerConfiguration.push_back(std::make_pair(layerName, shapeValues));
				break;
			}
		}
	}
}


ConfigParser::ConfigParser(std::string pathToFile)
{
	std::ifstream fileP(pathToFile);
	if (!fileP.is_open())
		throw std::invalid_argument("Cannot open network configuration file\n");
	std::string line;
	while (getline(fileP, line))
	{
		parseLine(line);
	}
}
