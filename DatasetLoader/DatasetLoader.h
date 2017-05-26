#ifndef DatasetLoader_h
#define DatasetLoader_h

#include "IDatasetLoader.h"
#include <common/Shape.h>
#include <common/Factory.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tokenizer.hpp>

namespace DatasetLoader {

template<typename T>
std::vector<T> oneHotEncode(size_t classIndex, size_t numClasses)
{
  // constructor initializes values to zeros
  std::vector<T> values(numClasses);
  values.at(classIndex) = 1;
  return values;
}

void splitLine(const std::string& line, std::vector<float> &values)
{
  values.clear();
  typedef boost::tokenizer<boost::char_separator<char>> tok_t;
  boost::char_separator<char> sep(" \t");
  tok_t tok(line, sep);
  values.reserve(std::distance(tok.begin(), tok.end()));
  std::transform(tok.begin(), tok.end(), std::back_inserter(values), [](const std::string val) { return std::stof(val); });
}

// data structure used to pass parameters to any/all dataset loader classes
struct DatasetLoaderBaseParams 
{
  const std::vector<std::string> & inputFiles_;
  const std::vector<std::string> & labelFiles_;
  const unsigned int batchSize_;
  DatasetLoaderBaseParams(const std::vector<std::string> & inputFiles, const std::vector<std::string> & outputFiles, const unsigned int batchSize) :
    inputFiles_(inputFiles), labelFiles_(outputFiles), batchSize_(batchSize) {};
};

// dataset loader implementation
template <typename InputDataType, typename LabelDataType>
class DatasetLoader : public IDatasetLoader
{
  typedef InputDataType T1;
  typedef LabelDataType T2;

private:
  // containers for inputs and outputs - every example and their according label set are vectors
  std::vector<std::vector<T1>> m_Inputs;
  std::vector<std::vector<T2>> m_Outputs;

  // iterators used for creating batches
  typename std::vector<std::vector<T1>>::iterator m_InputBatchIterator;
  typename std::vector<std::vector<T2>>::iterator m_OutputBatchIterator;

  bool m_IteratorsInitialized = false;

protected:
  NetworkConfiguration::Shape m_LearningExampleShape;
  NetworkConfiguration::Shape m_LabelShape;

  // number of examples per batch; defaults to whole dataset
  unsigned int m_BatchSize;

  // constructor is protected to avoid misuse - this class should not be used without inheritance
  DatasetLoader(unsigned int batchSize)
  {
    // if zero (no batches), set to maximum value
    m_BatchSize = (batchSize == 0) ? static_cast<unsigned int>(-1) : batchSize;
    // for dataset shuffling
    std::srand(unsigned(std::time(0)));
  }

  DatasetLoader(const DatasetLoaderBaseParams & params) : DatasetLoader(params.batchSize_) {};

  template<typename InputIterator>
  void addLearningExample(InputIterator first, InputIterator last)
  { 
    m_Inputs.push_back(std::vector<T1>(first, last));
    m_IteratorsInitialized = false;
  }

  template<typename InputIterator>
  void addLabel(InputIterator first, InputIterator last)
  {
    m_Outputs.push_back(std::vector<T2>(first, last));
    m_IteratorsInitialized = false;
  }

  // reserve space if number of training examples is known a priori 
  void reserveSpace(const unsigned int numExamples)
  {
    m_Inputs.reserve(numExamples);
    m_Outputs.reserve(numExamples);
    m_IteratorsInitialized = false;
  }

  void reserveInputSpace(const unsigned int numExamples)
  {
    m_Inputs.reserve(numExamples);
    m_IteratorsInitialized = false;
  }

  void reserveLabelSpace(const unsigned int numExamples)
  {
    m_Outputs.reserve(numExamples);
    m_IteratorsInitialized = false;
  }


public:
  void shuffleDataset() override
  {
    // zip inputs and outputs together so they get shuffled in the same way
    typedef boost::tuple<std::vector<std::vector<T1>>::iterator, std::vector<std::vector<T2>>::iterator> IteratorTuple;
    typedef boost::zip_iterator<IteratorTuple> ZipIterator;
    std::random_shuffle(ZipIterator(IteratorTuple(m_Inputs.begin(), m_Outputs.begin())), ZipIterator(IteratorTuple(m_Inputs.end(), m_Outputs.end())));
  }

  // returns false if it has iterated through the whole dataset
  bool nextBatch(tensorflow::Tensor& inputs, tensorflow::Tensor& expectedOutputs) override
  {
    // check if dataset can be used (iterators initialized and whole dataset has not been iterated through
    if (!readyForUse())
      return false;

    // batch size should be constant because of gradient calculation (see tensorflow::ops::Conv2DBackprop* documentation, there so no convenient way to calculate current batch size)
    if (std::distance(m_InputBatchIterator, m_Inputs.end()) < m_BatchSize || std::distance(m_OutputBatchIterator, m_Outputs.end()) < m_BatchSize)
    {
      m_InputBatchIterator = m_Inputs.end();
      m_OutputBatchIterator = m_Outputs.end();
      return false;
    }

    auto nextInputBatchIterator = m_InputBatchIterator + m_BatchSize;
    auto nextOutputBatchIterator = m_OutputBatchIterator + m_BatchSize;


    const unsigned int numExamples = m_BatchSize;

    // create new tensors
    NetworkConfiguration::Shape inputBatchShape { numExamples };
    inputBatchShape.insert(inputBatchShape.end(), m_LearningExampleShape.begin(), m_LearningExampleShape.end());
    NetworkConfiguration::Shape outputBatchShape { numExamples };
    outputBatchShape.insert(outputBatchShape.end(), m_LabelShape.begin(), m_LabelShape.end());
    inputs = tensorflow::Tensor(tensorflow::DT_FLOAT, inputBatchShape.asTensorShape());
    expectedOutputs = tensorflow::Tensor(tensorflow::DT_FLOAT, outputBatchShape.asTensorShape());

    // fill tensors with values
    auto inputsTensorIterator = inputs.flat<float>().data();
    auto outputsTensorIterator = expectedOutputs.flat<float>().data();
    std::for_each(m_InputBatchIterator, nextInputBatchIterator, [&inputsTensorIterator](const std::vector<T1>& example) {inputsTensorIterator = std::copy(example.begin(), example.end(), inputsTensorIterator);});
    std::for_each(m_OutputBatchIterator, nextOutputBatchIterator, [&outputsTensorIterator](const std::vector<T2>& exampleOutput) {outputsTensorIterator = std::copy(exampleOutput.begin(), exampleOutput.end(), outputsTensorIterator);});

    m_InputBatchIterator = nextInputBatchIterator;
    m_OutputBatchIterator = nextOutputBatchIterator;

    return true;
  }

  // resets dataset iterator so batching starts from the beginning - always call before exiting constructor of the derived class
  void resetBatchIterator() override
  {
    m_IteratorsInitialized = true;
    m_InputBatchIterator = m_Inputs.begin();
    m_OutputBatchIterator = m_Outputs.begin();
  }

  // returns true if iterators are initialized and they are not pointing at the end of the containers
  bool readyForUse() const override
  {
    return m_IteratorsInitialized && m_InputBatchIterator != m_Inputs.end() && m_OutputBatchIterator != m_Outputs.end();
  }
};

typedef std::function<IDatasetLoader*(DatasetLoaderBaseParams &)> DatasetLoaderCreator;
typedef Common::Factory<IDatasetLoader, std::string, DatasetLoaderCreator> DatasetLoaderFactory;

}

#endif
