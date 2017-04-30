#ifndef MNISTDatasetLoader_h
#define MNISTDatasetLoader_h

#include "DatasetLoader.h"
#include <fstream>

namespace DatasetLoader {

// loader for MNIST dataset (http://yann.lecun.com/exdb/mnist/) - loads images as tensors [height, width, channels]
class MNISTDatasetLoader : public DatasetLoader<unsigned char, unsigned char>
{
  bool readImageFile(std::string imageFilePath);
  bool readLabelFile(std::string labelFilePath);

public:
  MNISTDatasetLoader(std::string imageFilePath, std::string labelFilePath, unsigned int batchSize=0);
  MNISTDatasetLoader(const DatasetLoaderBaseParams & params) : MNISTDatasetLoader(params.inputFiles_.at(0), params.labelFiles_.at(0), params.batchSize_) {};
};

}

#endif
