#ifndef DatasetLoader_h
#define DatasetLoader_h

#include "IDatasetLoader.h"
#include <NetworkConfiguration/Shape.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <boost/iterator/zip_iterator.hpp>

namespace DatasetLoader {

// dataset loader implementation - template typenames are types of container used for inputs and outputs - they must have begin and end methods implemented
template <typename T1, typename T2>
class DatasetLoader : public IDatasetLoader
{
private:
  // containers for inputs and outputs
  std::vector<T1> m_Inputs;
  std::vector<T2> m_Outputs;

  // iterators used for creating batches
  typename std::vector<T1>::iterator m_InputBatchIterator;
  typename std::vector<T2>::iterator m_OutputBatchIterator;


protected:
	NetworkConfiguration::Shape m_InputShape;
	NetworkConfiguration::Shape m_OutputShape;

	// number of examples per batch; defaults to whole dataset
	unsigned int m_BatchSize;

	// constructor is protected to avoid misuse
	DatasetLoader(unsigned int batchSize=0)
	{
		// if zero (no batches), set to maximum value
		m_BatchSize = (batchSize == 0) ? static_cast<unsigned int>(-1) : batchSize;
	}

	/// setting inputs and expected outputs should also set batching iterators
  template<typename InputIterator>
	void setInputs(InputIterator first, InputIterator last)
	{
    m_Inputs.clear(); 
    m_Inputs.reserve(std::distance(first, last));
    m_Inputs.insert(m_Inputs.end(), first, last);
    // set iterator for batching
    m_InputBatchIterator = m_Inputs.begin();
	}

  template<typename InputIterator>
  void setOutputs(InputIterator first, InputIterator last)
  {
    m_Outputs.clear();
    m_Outputs.reserve(std::distance(first, last));
    // set iterator for batching
    m_Outputs.insert(m_Outputs.end(), first, last);
  }


public:
	void shuffleDataset() override
	{
		std::srand(unsigned(std::time(0)));
		// zip inputs and outputs together so they get shuffled in the same way
		typedef boost::tuple<std::vector<T1>::iterator, std::vector<T2>::iterator> IteratorTuple;
		typedef boost::zip_iterator<IteratorTuple> ZipIterator;
		std::random_shuffle(ZipIterator(IteratorTuple(m_Inputs.begin(), m_Outputs.begin())), ZipIterator(IteratorTuple(m_Inputs.end(), m_Outputs.end())));
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
