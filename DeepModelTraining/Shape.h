#include <vector>
#include <algorithm>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

class Shape
{
	// this class describes tensor (matrix) shape
private:
	// dimensions do not have to be defined - their value can be set to zero
	std::vector<tensorflow::int64> m_Values;
public:	
	template <class InputInterator>
	Shape(InputIterator valuesFirst, InputIterator valuesLast) : m_Values(valuesFirst, valuesLast) {}
	Shape(std::initializer_list<tensorflow::int64> list) : m_Values(list) {}
	Shape() : m_Values() {}
	void transpose() { std::reverse(m_Values.begin(), m_Values.end()); }
	size_t length() { return m_Values.size(); }
	bool validForParameterizedUse() 
	{
		// shape is valid for use with parameterized layers if all values in shape are greater than zero
		return std::find_if(m_Values.begin(), m_Values.end(), [](int n) { return n <= 0; }) == m_Values.end();
	
	}
	bool compatibleForMul(const Shape &right)
	{
		return m_Values.back() == right.m_Values.front();
	}
	tensorflow::TensorShape asTensorShape() { return tensorflow::TensorShape(tensorflow::gtl::ArraySlice<tensorflow::int64>(m_Values)); }
};	

