#include <vector>
#include <algorithm>
#include <numeric>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using tensorflow::int64;

class Shape
{
	// this class describes tensor (matrix) shape - implemented as a wrapper over std::vector<int64>
	std::vector<int64> m_Values;

public:
	template <class InputInterator>
	Shape(InputIterator valuesFirst, InputIterator valuesLast) : m_Values(valuesFirst, valuesLast) {}
	Shape(std::initializer_list<tensorflow::int64> list) : m_Values(list) {}
	Shape() : m_Values() {}
	void transpose() { std::reverse(m_Values.begin(), m_Values.end()); }
	std::vector<int64>::iterator begin() { return m_Values.begin(); }
	std::vector<int64>::iterator end() { return m_Values.end(); }
	std::vector<int64>::const_iterator cbegin() const { return m_Values.cbegin(); }
	std::vector<int64>::const_iterator cend() const { return m_Values.cend(); }
	int64 front() const { return m_Values.front(); }
	int64 back() const { return m_Values.back(); }
	size_t size() const { return m_Values.size(); }
	size_t numberOfElements() const { return std::accumulate(m_Values.begin(), m_Values.end(), 1, std::multiplies<int64>()); }
	void push_back(const int64& value) { m_Values.push_back(value); }
	template <class InputInterator>
	std::vector<int64>::iterator insert(std::vector<int64>::iterator pos, InputIterator first, InputIterator last) { return m_Values.insert(pos, first, last); }
	bool validForParameterizedUse() const
	{
		// shape is valid for use with parameterized layers if all values in shape are greater than zero
		return std::find_if(m_Values.cbegin(), m_Values.cend(), [](int n) { return n <= 0; }) == m_Values.cend();
	
	}
	bool compatibleForMul(const Shape &right) const
	{
		return this->back() == right.front();
	}
	tensorflow::TensorShape asTensorShape() const { return tensorflow::TensorShape(tensorflow::gtl::ArraySlice<int64>(this->m_Values)); }

	friend std::ostream& operator<< (std::ostream& os, const Shape &source)
	{
		os << source.m_Values;
		return os;
	}
};	

