#include <vector>

namespace Layer {

class TensorShape
{
private:
	std::vector<int> m_Values;
	// one value (first or last) in shape can be unspecifed (e.g., number of examples does not have to defined for creation of layers of network
	bool m_ValueUndefined;
	bool m_Transposed = false;
public:	
	template <class InputInterator>
	TensorShape(InputIterator valuesFirst, InputIterator valuesLast, bool firstValueUndefined) : m_Values(valuesFirst, valuesLast) { m_ValueUndefined = firstValueUndefined; }
	void transpose() { m_Transposed = !m_Transposed; }
	int length() { return (m_ValueUndefined) ? m_Values.size() + 1 : m_Values.size();  }
	bool validForParameterizedUse() 
	{
		// TODO: check if all values are not zero and everything is defined
	
	}
	bool compatibleForMul(const TensorShape &other)
	{
		/* TODO: check shapes and allow undefined values if possible */
		return true;
	}
};	

}
