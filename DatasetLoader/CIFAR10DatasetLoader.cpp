#include "CIFAR10DatasetLoader.h"

#define NUM_EXAMPLES 10000
#define HEIGHT 32
#define WIDTH 32
#define CHANNELS 3
#define NUM_LABELS 10

// for convinience, training set mean and std per channel was calculated outside, via python script
#define C_MEAN 83.886077
#define RC_STD 69.026443
#define GC_STD 69.024284
#define BC_STD 69.026657

namespace DatasetLoader {
  
bool CIFAR10DatasetLoader::readImageFile(std::string imageFilePath)
{
  std::ifstream file(imageFilePath, std::ios::binary);
  if (!file.is_open())
    return false;
  // set shapes - hardcoded
  m_LearningExampleShape = NetworkConfiguration::Shape({ HEIGHT, WIDTH, CHANNELS });
  m_LabelShape = NetworkConfiguration::Shape({ NUM_LABELS });
  const unsigned int exampleSize = HEIGHT * WIDTH * CHANNELS;
  const unsigned int channelSize = HEIGHT * WIDTH;
  unsigned char label;
  // allocate memory for the image file block
  unsigned char * memblock = nullptr;
  try
  {
    memblock = new unsigned char[exampleSize];
  }
  catch (...)
  {
    throw std::logic_error("Failed to allocate memory block for reading the dataset.");
  }
  std::vector<float> values;
  values.reserve(exampleSize);
  for (unsigned int i = 0; i < NUM_EXAMPLES; ++i)
  {
    // read label
    file.read((char*)&label, sizeof(label));
    std::vector<unsigned char> labelVector = oneHotEncode<unsigned char>(label, NUM_LABELS);
    addLabel(labelVector.begin(), labelVector.end());
    // read image data
    values.clear();
    file.read((char*)memblock, exampleSize);
    // normalize the values 
    std::transform(memblock, memblock + channelSize, std::back_inserter(values), [](unsigned char & val) {return (static_cast<float>(val) - C_MEAN) / RC_STD;});
    std::transform(memblock + channelSize, memblock + 2 * channelSize, std::back_inserter(values), [](unsigned char & val) {return (static_cast<float>(val) - C_MEAN) / GC_STD;});
    std::transform(memblock + 2 * channelSize, memblock + 3 * channelSize, std::back_inserter(values), [](unsigned char & val) {return (static_cast<float>(val) - C_MEAN) / BC_STD;});
    addLearningExample(values.begin(), values.end());
  }
  // all done
  delete[] memblock;
  file.close();
  return true;
}

CIFAR10DatasetLoader::CIFAR10DatasetLoader(std::vector<std::string> filePaths, unsigned int batchSize) : DatasetLoader(batchSize)
{
  // training dataset is composed of 5 ~30 MB files with 10000 examples each (labels included)
  for (auto it = filePaths.begin(); it != filePaths.end(); ++it)
  {
    const std::string & imageFilePath = *it;
    if (!readImageFile(imageFilePath))
      throw std::logic_error("Failed while opening image file.");
  }
  resetBatchIterator();
}

}

// register class in factory
namespace {
  using namespace DatasetLoader;
  DatasetLoaderCreator ctor = [](DatasetLoaderBaseParams & params) {return new CIFAR10DatasetLoader(params);};
  bool dummy = DatasetLoaderFactory::instance().registerClass("CIFAR10DatasetLoader", ctor);
}
