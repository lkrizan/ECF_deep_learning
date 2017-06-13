#include "CIFAR10DatasetLoader.h"

#define NUM_EXAMPLES 10000
#define HEIGHT 32
#define WIDTH 32
#define CHANNELS 3
#define NUM_LABELS 10

// for convinience, training set mean and std per channel was calculated outside, via python script
#define RC_MEAN 125.3069
#define GC_MEAN 122.950149
#define BC_MEAN 113.865997
#define RC_STD 62.993251
#define GC_STD 62.088603
#define BC_STD 66.705009

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
  std::vector<float> rChannel;
  std::vector<float> gChannel;
  std::vector<float> bChannel;
  rChannel.reserve(channelSize);
  gChannel.reserve(channelSize);
  bChannel.reserve(channelSize);
  for (unsigned int i = 0; i < NUM_EXAMPLES; ++i)
  {
    // read label
    file.read((char*)&label, sizeof(label));
    std::vector<unsigned char> labelVector = oneHotEncode<unsigned char>(label, NUM_LABELS);
    addLabel(labelVector.begin(), labelVector.end());
    // read image data
    values.clear();
    rChannel.clear();
    gChannel.clear();
    bChannel.clear();
    file.read((char*)memblock, exampleSize);
    // normalize the values
    std::transform(memblock, memblock + channelSize, std::back_inserter(rChannel), [](unsigned char & val) {return (static_cast<float>(val) - RC_MEAN) / RC_STD;});
    std::transform(memblock + channelSize, memblock + 2 * channelSize, std::back_inserter(gChannel), [](unsigned char & val) {return (static_cast<float>(val) - GC_MEAN) / GC_STD;});
    std::transform(memblock + 2 * channelSize, memblock + 3 * channelSize, std::back_inserter(bChannel), [](unsigned char & val) {return (static_cast<float>(val) - BC_MEAN) / BC_STD;});
    for (unsigned int i = 0; i < channelSize; ++i)
    {
      values.push_back(rChannel[i]);
      values.push_back(gChannel[i]);
      values.push_back(bChannel[i]);
    }
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
