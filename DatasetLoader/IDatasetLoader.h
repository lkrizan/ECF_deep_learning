#ifndef IDatasetLoader_h
#define IDatasetLoader_h

#include <tensorflow/core/framework/tensor.h> 

namespace DatasetLoader {

// common interface 

class IDatasetLoader
{
public:
	virtual ~IDatasetLoader() = default;
	virtual void shuffleDataset() = 0;
	// returns false if it has iterated through the whole dataset
	virtual bool nextBatch(tensorflow::Tensor& inputs, tensorflow::Tensor& expectedOutputs) = 0;
	// resets dataset iterator so batching starts from the beginning
	virtual void resetBatchIterator() = 0;
};

typedef std::shared_ptr<IDatasetLoader> IDatasetLoaderP;

}	// namespace DatasetLoader

#endif
