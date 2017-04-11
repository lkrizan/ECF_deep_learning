#include "ConfigParser.h"
#include <sstream>

void ConfigParser::parseLine(const std::string line)
{
	typedef boost::tokenizer<boost::char_separator<char>> tok_t;
	boost::char_separator<char> sep(" ");
	tok_t tokens(line, sep);
	if (tokens.begin() != tokens.end())
	{
		// handle comments
		if (*tokens.begin() == "#")
			return;
		auto currIterator = tokens.begin();
		if (isHeaderLine(*currIterator))
		{
			std::string header((*currIterator).begin() + 1, (*currIterator).end() - 1);
			// this is a header line - determine next state
			if (header == "General")
				m_State = eGeneral;
			else if (header == "Layers")
				m_State = eLayers;
			else if (header == "Loss")
				m_State = eLoss;
			else if (header == "Save")
				m_State = eSave;
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
					{
						m_NumInputs = std::stoi(*currIterator);
						inputsConfigured = true;
					}
					else
						throw std::logic_error(errorMsg);
				}
				else if (*currIterator == "NumOutputs")
				{
					if (++currIterator != tokens.end())
					{
						m_NumOutputs = std::stoi(*currIterator);
						outputsConfigured = true;
					}
					else
						throw std::logic_error(errorMsg);
				}
				else if (*currIterator == "DatasetPath")
				{
					if (++currIterator != tokens.end())
					{
						m_DatasetPath = *currIterator;
						datasetPathConfigured = true;
					}
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
				layerConfigurationConfigured = true;
				break;
			}
			case eLoss:
				m_LossFunctionName = *currIterator;
				lossFunctionConfigured = true;
				m_State = eLossFinished;
				break;
			case eLossFinished:
				throw std::logic_error("Only one loss function is allowed.\n");
				break;
			case eSave:
			{
				// optional parameters
				m_SaveModel = true;
				if (*currIterator == "SavePath")
				{
					if (++currIterator != tokens.end())
					{
						m_SaveFolderPath = *currIterator;
					}
				}
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

	// create error messages (if errors exist)
	bool parameterizationFailure = !(inputsConfigured && outputsConfigured && datasetPathConfigured && layerConfigurationConfigured && lossFunctionConfigured);
	std::stringstream errorMessageStream;
	if (!inputsConfigured)
	{
		errorMessageStream << "Input shape must be declared." << std::endl;
	}
	if (!outputsConfigured)
	{
		errorMessageStream << "Output shape must be declared." << std::endl;
	}
	if (!datasetPathConfigured)
	{
		errorMessageStream << "Dataset path is missing." << std::endl;
	}
	if (!layerConfigurationConfigured)
	{
		errorMessageStream << "Layer configuration is missing." << std::endl;
	}
	if (!lossFunctionConfigured)
	{
		errorMessageStream << "Loss function must be declared." << std::endl;
	}

	if (parameterizationFailure)
	{
		throw std::logic_error(errorMessageStream.str());
	}
}
