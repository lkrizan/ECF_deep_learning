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
  typename IdentifierType                     // key type (e.g., std::string)
>
class Factory
{
  // map that holds object type and its constructor
  typedef std::function<AbstractProduct*()> AbstractCreator;
  typedef std::map<IdentifierType, AbstractCreator> AssocMap;
  AssocMap m_AssocMap;
  
  // make constructors and operator= private so it cannot be instanced
  Factory() = default;
  Factory(const Factory &) = delete;
  Factory & operator=(const Factory &) = delete;

public:
  static Factory & instance() { static Factory f; return f; }

  template<typename ProductCreator>
  void registerClass(const IdentifierType& id, ProductCreator creator)
  {
    m_AssocMap.insert(std::pair<IdentifierType, AbstractCreator>(id, creator));
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

template
<
  typename AbstractProduct,
  typename ManufacturedType,
  typename IdentifierType=std::string
>

class RegisterInFactory
{
public:
  static AbstractProduct* CreateInstance()
  {
    return new ManufacturedType();
  }

  RegisterInFactory(const IdentifierType &id)
  {
    Factory<AbstractProduct, IdentifierType>::instance().registerClass(id, CreateInstance);
  }

};


}   // namespace Common

#endif
