#ifndef CIFAR10DatasetReader_h
#define CIFAR10DatasetReader_h

#include "DatasetLoader.h"
#include <fstream>

namespace DatasetLoader {


// loader for CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) - loads images as tensors [height, width, channels] (32, 32, 3)
class CIFAR10DatasetLoader : public DatasetLoader<float, unsigned char>
{
  bool readImageFile(std::string imageFilePath);

public:
  CIFAR10DatasetLoader(std::vector<std::string> filePaths, unsigned int batchSize=0);
  CIFAR10DatasetLoader(const DatasetLoaderBaseParams & params) : CIFAR10DatasetLoader(params.inputFiles_, params.batchSize_) {};
};


}

#endif