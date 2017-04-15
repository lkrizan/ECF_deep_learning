#ifndef DatasetLoader_h
#define DatasetLoader_h

#include <tensorflow/core/framework/tensor.h> 

namespace DatasetLoader {

// abstract dataset loader
class DatasetLoader
{
protected:
	virtual void shuffleDataset() {};
public:
	virtual ~DatasetLoader() {};
	// returns false if it has iterated through the whole dataset
	virtual bool nextBatch(tensorflow::Tensor& inputs, tensorflow::Tensor& expectedOutputs) = 0;
	// resets dataset iterator so batching starts from the beginning
	virtual void resetBatchIterator() = 0;
};

typedef std::shared_ptr<DatasetLoader> DatasetLoaderP;

}

#endif
