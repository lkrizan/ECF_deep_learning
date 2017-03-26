#include <vector>
#include <algorithm>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

class Shape : public std::vector<tensorflow::int64>
{
	// this class describes tensor (matrix) shape
public:	
	template <class InputInterator>
	Shape(InputIterator valuesFirst, InputIterator valuesLast) : std::vector<tensorflow::int64>(valuesFirst, valuesLast) {}
	Shape(std::initializer_list<tensorflow::int64> list) : std::vector<tensorflow::int64>(list) {}
	Shape() : std::vector<tensorflow::int64>() {}
	void transpose() { std::reverse(this->begin(), this->end()); }
	bool validForParameterizedUse() 
	{
		// shape is valid for use with parameterized layers if all values in shape are greater than zero
		return std::find_if(this->begin(), this->end(), [](int n) { return n <= 0; }) == this->end();
	
	}
	bool compatibleForMul(const Shape &right)
	{
		return this->back() == right.front();
	}
	tensorflow::TensorShape asTensorShape() { return tensorflow::TensorShape(tensorflow::gtl::ArraySlice<tensorflow::int64>(*this)); }
};	

