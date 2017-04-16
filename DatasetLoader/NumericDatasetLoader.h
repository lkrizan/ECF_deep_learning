#ifndef NumericDatasetLoader_h
#define NumericDatasetLoader_h

#include "DatasetLoader.h"
#include <boost/tokenizer.hpp>
#include <fstream> 

namespace DatasetLoader {

/*
*	This class is used for reading / handling numeric datasets. Both inputs and outputs are part of the same file in the following format:
	number_of_inputs number_of_outputs							/// first line
	input1 input2 ... inputn output1 output2 ... outputm		/// all other lines
	
	All values are saved and returned as float. Learning examples are limited to vectors (matrices and tensors cannot be used as learning examples with this loader).

*/

class NumericDatasetLoader : public DatasetLoader<std::vector<float>, std::vector<float>>
{
	void parseLine(const std::string& line, std::vector<float> &values) const;

public:
	NumericDatasetLoader(const std::string datasetPath, const unsigned int batchSize = 0);
};

}	// namespace DatasetLoader

#endif
