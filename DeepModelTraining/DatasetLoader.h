#include <vector>
#include <fstream>
#include <boost/tokenizer.hpp>

template <class T>
class DatasetLoader{
public:
    DatasetLoader(const std::string fileName, const int numInputs, const int numOutputs);
    std::vector<T>& getInputs();
    std::vector<T>& getOutputs();
private:
    std::vector<T> m_Inputs;
    std::vector<T> m_Outputs;
    void parseLine(const std::string &line, std::vector<T> &values);
};