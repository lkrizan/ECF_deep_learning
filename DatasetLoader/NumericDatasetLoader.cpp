#include "NumericDatasetLoader.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>

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

void NumericDatasetLoader::shuffleDataset()
{
	std::srand(unsigned(std::time(0)));
	int randg = std::rand();
	// ensure that both inputs and expected outputs are shuffled the same way
	auto randomMember = [&randg](const int i) { return randg % i; };
	std::random_shuffle(m_Inputs.begin(), m_Inputs.end(), randomMember);
	std::random_shuffle(m_Outputs.begin(), m_Outputs.end(), randomMember);
}

NumericDatasetLoader::NumericDatasetLoader(const std::string datasetPath, const unsigned int batchSize)
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
		m_NumInputs = static_cast<unsigned int>(values[0]);
		m_NumOutputs = static_cast<unsigned int>(values[1]);
	}
	// read all other values
	while (getline(fileP, line))
	{
		parseLine(line, values);
		m_Inputs.insert(m_Inputs.end(), values.begin(), values.begin() + m_NumInputs);
		m_Outputs.insert(m_Outputs.end(), values.begin() + m_NumInputs, values.end());
	}
	fileP.close();

	// initialize iterators for batching
	m_InputBatchIterator = m_Inputs.begin();
	m_OutputBatchIterator = m_Outputs.begin();

	// if batch size is set to zero, there is no splitting into batches
	m_BatchSize = (batchSize > 0) ? batchSize : m_Inputs.size() / m_NumInputs;
}

bool NumericDatasetLoader::nextBatch(tensorflow::Tensor & inputs, tensorflow::Tensor & expectedOutputs)
{
	// check if whole dataset has been used
	if (m_InputBatchIterator == m_Inputs.end() || m_OutputBatchIterator == m_Outputs.end())
		return false;

	// calculate end iterator for inputs and outputs in this batch
	unsigned int inputsStride = m_NumInputs * m_BatchSize;
	unsigned int outputsStride = m_NumOutputs * m_BatchSize;
	auto nextInputBatchIterator = (std::distance(m_InputBatchIterator, m_Inputs.end()) > inputsStride) ? m_InputBatchIterator + inputsStride : m_Inputs.end();
	auto nextOutputBatchIterator = (std::distance(m_OutputBatchIterator, m_Outputs.end()) > outputsStride) ? m_OutputBatchIterator + outputsStride : m_Outputs.end();

	// number of examples contained in the batch
	unsigned int numExamples = std::distance(m_InputBatchIterator, nextInputBatchIterator) / m_NumInputs;

	// create tensors new tensors (ownership is handeled by caller)
	inputs = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ numExamples, m_NumInputs }));
	expectedOutputs = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ numExamples, m_NumOutputs }));

	// fill tensors with values
	std::copy(m_InputBatchIterator, nextInputBatchIterator, inputs.flat<float>().data());
	std::copy(m_OutputBatchIterator, nextOutputBatchIterator, expectedOutputs.flat<float>().data());

	m_InputBatchIterator = nextInputBatchIterator;
	m_OutputBatchIterator = nextOutputBatchIterator;

	return true;
}

void NumericDatasetLoader::resetBatchIterator()
{
	m_InputBatchIterator = m_Inputs.begin();
	m_OutputBatchIterator = m_Outputs.begin();
}


}	// namespace DatasetLoader
