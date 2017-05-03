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
      if (header == "Dataset")
        m_State = eDataset;
      else if (header == "Layers")
        m_State = eLayers;
      else if (header == "Loss")
        m_State = eLoss;
      else
        throw std::logic_error(*currIterator + " is not registered block in configuration file.\n");
      return;
    }

    switch (m_State)
    {
      case eStart:
        throw std::logic_error("Parameters cannot be declared outside of blocks in configuration file.\n");
        break;
      case eDataset:
      {
        std::string errorMsg = "Parameters in general block require argument.\n";
        if (*currIterator == "loader")
        {
          if (++currIterator != tokens.end())
          {
            m_DatasetLoaderType = *currIterator;
            datasetLoaderTypeConfigured = true;
          }
          else
            throw std::logic_error(errorMsg);
        }
        else if (*currIterator == "InputShape")
        {
          if (++currIterator != tokens.end())
          {
            m_InputShape.reserve(std::distance(currIterator, tokens.end()));
            std::transform(currIterator, tokens.end(), std::back_inserter(m_InputShape), [](const std::string val) { return std::stof(val); });
            inputsConfigured = true;
          }
          else
            throw std::logic_error(errorMsg);
        }
        else if (*currIterator == "OutputShape")
        {
          if (++currIterator != tokens.end())
          {
            m_OutputShape.reserve(std::distance(currIterator, tokens.end()));
            std::transform(currIterator, tokens.end(), std::back_inserter(m_OutputShape), [](const std::string val) { return std::stof(val); });
            outputsConfigured = true;
          }
          else
            throw std::logic_error(errorMsg);
        }
        else if (*currIterator == "inputFiles")
        {
          if (++currIterator != tokens.end())
          {
            m_InputFiles.reserve(std::distance(currIterator, tokens.end()));
            m_InputFiles.insert(m_InputFiles.end(), currIterator, tokens.end());
            if (m_InputFiles.size() >= 1)
              datasetPathConfigured = true;
          }
          else
            throw std::logic_error(errorMsg);
        }

        // optional
        else if (*currIterator == "labelFiles")
        {
          if (++currIterator != tokens.end())
          {
            m_LabelFiles.reserve(std::distance(currIterator, tokens.end()));
            m_LabelFiles.insert(m_LabelFiles.end(), currIterator, tokens.end());
          }
        }

        else if (*currIterator == "batchSize")
        {
          if (++currIterator != tokens.end())
            m_BatchSize = std::stoi(*currIterator);
        }
        else
          throw std::logic_error(*currIterator + " is not a valid parameter in General block in configuration file.\n");
        break;
      }
      case eLayers:
      {
        std::vector<std::vector<int>> layerArguments;
        std::string layerName = *currIterator;
        // iterate through all argumets (they are tokens, sparated by comma ','
        boost::char_separator<char> commaSep(",");
        while (++currIterator != tokens.end())
        {
          tok_t shapeTokens(*currIterator, commaSep);
          std::vector<int> values;
          values.reserve(std::distance(shapeTokens.begin(), shapeTokens.end()));
          std::transform(shapeTokens.begin(), shapeTokens.end(), std::back_inserter(values), [](const std::string& val) {return std::stoi(val);});
          layerArguments.push_back(values);
        }
        m_LayerConfiguration.push_back(std::make_pair(layerName, layerArguments));
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
  if (!datasetLoaderTypeConfigured)
  {
    errorMessageStream << "Dataset loader type must be declared." << std::endl;
  }

  if (parameterizationFailure)
  {
    throw std::logic_error(errorMessageStream.str());
  }
}
