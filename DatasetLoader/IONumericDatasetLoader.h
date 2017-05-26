#ifndef IONumericDatasetLoader_h
#define IONumericDatasetLoader_h

#include "DatasetLoader.h"
#include <fstream> 

namespace DatasetLoader {
/*
Same as NumericDatasetLoader, but reads inputs and outputs from separate files.
*/
class IONumericDatasetLoader : public DatasetLoader<float, float>
{
public:
  // the great anti-pattern strikes again - everything is done through the constructor and that's about it
  IONumericDatasetLoader(const std::string inputsPath, const std::string labelsPath, const unsigned int batchSize = 0);
  IONumericDatasetLoader(DatasetLoaderBaseParams params) : IONumericDatasetLoader(params.inputFiles_.at(0), params.labelFiles_.at(0), params.batchSize_) {};
};

}
#endif
