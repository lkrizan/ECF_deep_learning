#ifndef IRNGenerator_h
#define IRNGenerator_h

#include <random>
#include <memory>
#include <functional>
#include <Common/Factory.h>

// Abstract interface for random number generator functors used for individual initialization
template <typename RealType>
class IRNGenerator
{
public:
  virtual ~IRNGenerator() = default;
  virtual RealType operator()() = 0;
};

template <typename RealType>
struct RNGBaseParams
{
  RealType firstArgument_;
  RealType secondArgument_;
  RNGBaseParams(RealType firstArgument, RealType secondArgument) : firstArgument_(firstArgument), secondArgument_(secondArgument) {}
};

template<typename RealType>
using RNGeneratorP = std::shared_ptr<IRNGenerator<RealType>>;

template<typename RealType>
using RNGCreator = std::function<IRNGenerator<RealType>* (RNGBaseParams<RealType> &)>;

// typedef Common::Factory<Layer, std::string, LayerCreator> LayerFactory;
template<typename RealType>
using RNGFactory = Common::Factory<IRNGenerator<RealType>, std::string, RNGCreator<RealType>>;




#endif