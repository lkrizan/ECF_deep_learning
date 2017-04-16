#ifndef DatasetLoader_h
#define DatasetLoader_h

#include "IDatasetLoader.h"
#include <NetworkConfiguration/Shape.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>

namespace DatasetLoader {

// dataset loader implementation - template typenames are types of container used for inputs and outputs - they must have begin and end methods implemented
template <typename T1, typename T2>
class DatasetLoader : public IDatasetLoader
{

protected:
	NetworkConfiguration::Shape m_InputShape;
	NetworkConfiguration::Shape m_OutputShape;

	// number of examples per batch; if set to zero, defaults to whole dataset
	unsigned int m_BatchSize;

	// containers for inputs and outputs
	std::vector<T1> m_Inputs;
	std::vector<T2> m_Outputs;

	// iterators used for creating batches
	typename std::vector<T1>::iterator m_InputBatchIterator;
	typename std::vector<T2>::iterator m_OutputBatchIterator;


public:
	void shuffleDataset() override
	{
		std::srand(unsigned(std::time(0)));
		int randg = std::rand();
		// ensure that both inputs and expected outputs are shuffled the same way
		auto randomMember = [&randg](const int i) { return randg % i; };
		std::random_shuffle(m_Inputs.begin(), m_Inputs.end(), randomMember);
		std::random_shuffle(m_Outputs.begin(), m_Outputs.end(), randomMember);
	}

	// returns false if it has iterated through the whole dataset
	bool nextBatch(tensorflow::Tensor& inputs, tensorflow::Tensor& expectedOutputs) override
	{
		// check if whole dataset has been used
		if (m_InputBatchIterator == m_Inputs.end() || m_OutputBatchIterator == m_Outputs.end())
			return false;

		auto nextInputBatchIterator = (std::distance(m_InputBatchIterator, m_Inputs.end()) > m_BatchSize) ? m_InputBatchIterator + m_BatchSize : m_Inputs.end();
		auto nextOutputBatchIterator = (std::distance(m_OutputBatchIterator, m_Outputs.end()) > m_BatchSize) ? m_OutputBatchIterator + m_BatchSize : m_Outputs.end();

		const unsigned int numExamples = std::distance(m_InputBatchIterator, nextInputBatchIterator);

		// create new tensors
		NetworkConfiguration::Shape inputBatchShape { numExamples };
		inputBatchShape.insert(inputBatchShape.end(), m_InputShape.begin(), m_InputShape.end());
		NetworkConfiguration::Shape outputBatchShape { numExamples };
		outputBatchShape.insert(outputBatchShape.end(), m_OutputShape.begin(), m_OutputShape.end());
		inputs = tensorflow::Tensor(tensorflow::DT_FLOAT, inputBatchShape.asTensorShape());
		expectedOutputs = tensorflow::Tensor(tensorflow::DT_FLOAT, outputBatchShape.asTensorShape());

		// fill tensors with values
		auto inputsTensorIterator = inputs.flat<float>().data();
		auto outputsTensorIterator = expectedOutputs.flat<float>().data();
		std::for_each(m_InputBatchIterator, nextInputBatchIterator, [&inputsTensorIterator](const T1& example) {inputsTensorIterator = std::copy(example.begin(), example.end(), inputsTensorIterator);});
		std::for_each(m_OutputBatchIterator, nextOutputBatchIterator, [&outputsTensorIterator](const T2& exampleOutput) {outputsTensorIterator = std::copy(exampleOutput.begin(), exampleOutput.end(), outputsTensorIterator);});

		m_InputBatchIterator = nextInputBatchIterator;
		m_OutputBatchIterator = nextOutputBatchIterator;

		return true;
	}

	// resets dataset iterator so batching starts from the beginning
	void resetBatchIterator() override
	{
		m_InputBatchIterator = m_Inputs.begin();
		m_OutputBatchIterator = m_Outputs.begin();
	}
};

}

#endif
