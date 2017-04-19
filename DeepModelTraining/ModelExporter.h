#ifndef ModelExporter_h
#define ModelExporter_h

#include <fstream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <NetworkConfiguration/Shape.h>
#include <ECF/ECF.h>

#define VARIABLE_FILE_NAME "variables.dat"
#define GRAPH_FILE_NAME "graph.pb"

class ModelExporter
{
  std::string m_FolderPath;
  std::ofstream m_OutputStream;


public:

  ModelExporter(StateP ECFState, std::string folderPath) : m_FolderPath(folderPath)
  { 
    m_OutputStream.open(m_FolderPath + VARIABLE_FILE_NAME);
    if (!m_OutputStream.is_open())
    {
      std::string msg = "Warning: failed to open export file: " + m_FolderPath + VARIABLE_FILE_NAME + ". Trying to save files to root directory...";
      ECF_LOG(ECFState, 2, msg);
      m_FolderPath = "";
      m_OutputStream.open(VARIABLE_FILE_NAME);
      if (!m_OutputStream.is_open())
        throw std::logic_error("Failed to create export file.");
    }
  }

  ~ModelExporter()
  {
    if (m_OutputStream.is_open())
      m_OutputStream.close();
  }

  void exportGraph(tensorflow::GraphDef graphDefinition)
  {
    tensorflow::WriteBinaryProto(tensorflow::Env::Default(), m_FolderPath + GRAPH_FILE_NAME, graphDefinition);
  }

  template <typename InputIterator>
  void exportVariableValues(std::string variableName, NetworkConfiguration::Shape shape, InputIterator firstValIt, InputIterator lastValIt)
  {
    /*
    Output format:
    tensor name - string \n
    tensor shape - integers separated with space \n
    values - integers separated with space \n
    new line
    */
    using tensorflow::int64;
    m_OutputStream << variableName << std::endl;
    m_OutputStream << shape << std::endl;
    for_each(firstValIt, lastValIt, [this](const float & val) {m_OutputStream << val << " ";});
    m_OutputStream << std::endl << std::endl;
  }
};

#endif
