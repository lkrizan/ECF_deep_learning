#include "MNISTDatasetLoader.h"
#include <stdlib.h>

#define NUM_CLASSES 10

namespace DatasetLoader {

bool MNISTDatasetLoader::readImageFile(std::string imageFilePath)
{
  std::ifstream file(imageFilePath, std::ios::binary);
  if (!file.is_open())
    return false;
  // read image file
  unsigned int tmp, numImages, numRows, numCols;
  // first entry in file is "the magic number" (4 bytes), which is not used here
  file.read((char*)&tmp, sizeof(tmp));
  // second entry is the number of images (4 bytes)
  file.read((char*)&tmp, sizeof(tmp));
  // dataset uses big endian format for integers, so its bytes need to be reversed
  numImages = _byteswap_ulong(tmp);
  // reserve space in dataset containers to improve performance
  reserveInputSpace(numImages);
  // number of rows (should be 28 for MNIST)
  file.read((char*)&tmp, sizeof(tmp));
  numRows = _byteswap_ulong(tmp);
  // number of columns (should be 28 for MNIST)
  file.read((char*)&tmp, sizeof(tmp));
  numCols = _byteswap_ulong(tmp);
  // set learning example shape
  m_LearningExampleShape = NetworkConfiguration::Shape({ numRows, numCols, 1 });

  // read actual images
  for (unsigned int i = 0; i < numImages; ++i)
  {
    std::vector<unsigned char> values;
    values.reserve(numRows * numCols);
    for (unsigned int j = 0; j < numCols * numRows; ++j)
    {
      unsigned char tmp;
      file.read((char*)&tmp, sizeof(tmp));
      values.push_back(tmp);
    }
    // set read values
    addLearningExample(values.begin(), values.end());
  }

  // all done
  return true;

}

bool MNISTDatasetLoader::readLabelFile(std::string labelFilePath)
{
  std::ifstream file(labelFilePath, std::ios::binary);
  if (!file.is_open())
    return false;
  // read lables file
  unsigned int tmp, numImages;
  // MNIST dataset unfortunately does not contain information about number of classes so its done this way
  // this may change with refactoring so this loader can be used with all datasets with the same format
  unsigned int numClasses = NUM_CLASSES;
  // first entry in file is "the magic number" (4 bytes), which is not used here
  file.read((char*)&tmp, sizeof(tmp));
  // second entry is the number of images (4 bytes)
  file.read((char*)&tmp, sizeof(tmp));
  // dataset uses big endian format for integers, so its bytes need to be reversed
  numImages = _byteswap_ulong(tmp);
  // reserve space for all labels
  reserveLabelSpace(numImages);
  // set label shape
  m_LabelShape = NetworkConfiguration::Shape({ numClasses });

  // read actual labels and convert them to one-hot encoded vectors
  for (unsigned int i = 0; i < numImages; ++i)
  {
    unsigned char tmp;
    file.read((char*)&tmp, sizeof(tmp));
    std::vector<unsigned char> values = oneHotEncode<unsigned char>(tmp, numClasses);
    addLabel(values.begin(), values.end());
  }

  // all done
  return true;


}

MNISTDatasetLoader::MNISTDatasetLoader(std::string imageFilePath, std::string labelFilePath, unsigned int batchSize) : DatasetLoader(batchSize)
{
  if (!readImageFile(imageFilePath))
    throw std::logic_error("Failed while opening image file.");
  if (!readLabelFile(labelFilePath))
    throw std::logic_error("Failed while opening label file.");
}

}
