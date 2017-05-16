#include "UniformDistributionRNG.h"

template <typename RealType>
RealType UniformDistributionRNG<RealType>::operator()()
{
  return m_Generator(m_RNEngine);
}

namespace {
  RNGCreator<float> ctorNormal1 = [](RNGBaseParams<float> & params) {return new UniformDistributionRNG<float>(params);};
  bool dummy1 = RNGFactory<float>::instance().registerClass("UniformDistributionRNG", ctorNormal1);
  RNGCreator<double> ctorNormal2 = [](RNGBaseParams<double> & params) {return new UniformDistributionRNG<double>(params);};
  bool dummy2 = RNGFactory<double>::instance().registerClass("UniformDistributionRNG", ctorNormal2);
}