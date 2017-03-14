#include "DatasetLoader.h"

template <class T>
std::vector<T>& DatasetLoader<T>::getInputs()
{
    return m_Inputs;
}

template <class T>
std::vector<T>& DatasetLoader<T>::getOutputs()
{
    return m_Outputs;
}

template <class T>
void DatasetLoader<T>::parseLine(const std::string &line, std::vector<T> &values)
{
    values.clear();
    typedef boost::tokenizer<boost::char_separator<char>> tok_t;
    boost::char_separator<char> sep(" ", "", boost::keep_empty_tokens);
    tok_t tok(line, sep);
    for (auto i = tok.begin(); i != tok.end(); i++)
        values.push_back((T) std::stod(*i));
}


template <class T>
DatasetLoader<T>::DatasetLoader(const std::string fileName, const int numInputs, const int numOutputs)
{
    std::ifstream fileP(fileName);
    if (!fileP.is_open())
        return;
    std::string line;
    while(getline(fileP, line))
    {
        std::vector<T> values;
        parseLine(line, values);
        // concatenate values to inputs and outputs
        m_Inputs.insert(m_Inputs.end(), values.begin(), values.begin() + numInputs);
        m_Outputs.insert(m_Outputs.end(), values.begin() + numInputs, values.end());
    }
    fileP.close();
}

template class DatasetLoader<float>;
template class DatasetLoader<double>;