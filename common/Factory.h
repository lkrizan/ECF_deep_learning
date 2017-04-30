#ifndef Factory_h
#define Factory_h

/*
Almost generic implementation of the factory pattern, as a singleton.
Inspired by A. Alexandrescu's implementation from the book 'Modern C++ Design' 
and H. Sutter's articles about factory pattern on http://www.drdobbs.com
Creating object returns a shared pointer to the object created.
*/

#include <map>
#include <memory>

namespace Common {

template 
<
  typename AbstractProduct,                   // base-class type
  typename IdentifierType,                    // key type (e.g., std::string)
  typename ProductCreator                     // object constructor
>
class Factory
{
  // map that holds object type and its constructor
  typedef std::map<IdentifierType, ProductCreator> AssocMap;
  AssocMap m_AssocMap;
  
  // make constructors and operator= private so it cannot be instanced
  Factory() = default;
  Factory(const Factory &) = delete;
  Factory & operator=(const Factory &) = delete;

public:
  static Factory & instance() { static Factory f; return f; }

  // returns true if registering succeeded
  bool registerClass(const IdentifierType& id, ProductCreator creator)
  {
    // insert returns pair<iterator, bool>
    return m_AssocMap.insert(std::pair<IdentifierType, ProductCreator>(id, creator)).second;
  }

  // returns true if erasing succeeded
  bool unregisterClass(const IdentifierType &id)
  {
    // this erase overload returns number of elements erased
    return m_AssocMap.erase(id) == 1;
  }

  template <typename ArgumentType>
  std::shared_ptr<AbstractProduct> createObject(const IdentifierType& id, ArgumentType & args)
  {
    typename AssocMap::const_iterator it = m_AssocMap.find(id);
    if (it != m_AssocMap.end())
    {
      return std::shared_ptr<AbstractProduct>((it->second) (args));
    }
    // error handling (unregistered factory)
    throw std::exception("Unknown object type passed to factory.");
  }

  std::shared_ptr<AbstractProduct> createObject(const IdentifierType& id)
  {
    typename AssocMap::const_iterator it = m_AssocMap.find(id);
    if (it != m_AssocMap.end())
    {
      return std::shared_ptr<AbstractProduct>((it->second) ());
    }
    // error handling (unregistered factory)
    throw std::exception("Unknown object type passed to factory.");
  }

};


}   // namespace Common

#endif
