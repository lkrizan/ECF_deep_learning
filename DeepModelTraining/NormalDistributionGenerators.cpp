#include "NormalDistributionGenerators.h"

template <typename RealType>
RealType NormalDistributionRNG<RealType>::operator()()
{
  return m_Generator(m_RNEngine);
}

template <typename RealType>
RealType TruncatedNormalDistributionRNG<RealType>::operator()()
{
  // if the generated number is outside of mean more than 2 values of stddev, try again
  while (true)
  {
    RealType number = m_Generator(m_RNEngine);
    if (number >= m_Mean - 2 * m_StdDev && number <= m_Mean + 2 * m_StdDev)
      return number;
  }
}

namespace {
  // register viable class templates in appropriate factories
  RNGCreator<float> ctorNormal1 = [](RNGBaseParams<float> & params) {return new NormalDistributionRNG<float>(params);};
  bool dummy1 = RNGFactory<float>::instance().registerClass("NormalDistributionRNG", ctorNormal1);
  RNGCreator<double> ctorNormal2 = [](RNGBaseParams<double> & params) {return new NormalDistributionRNG<double>(params);};
  bool dummy2 = RNGFactory<double>::instance().registerClass("NormalDistributionRNG", ctorNormal2);
  RNGCreator<float> ctorTruncated1 = [](RNGBaseParams<float> & params) {return new TruncatedNormalDistributionRNG<float>(params);};
  bool dummy3 = RNGFactory<float>::instance().registerClass("TruncatedNormalDistributionRNG", ctorTruncated1);
  RNGCreator<double> ctorTruncated2 = [](RNGBaseParams<double> & params) {return new TruncatedNormalDistributionRNG<double>(params);};
  bool dummy4 = RNGFactory<double>::instance().registerClass("TruncatedNormalDistributionRNG", ctorTruncated2);
}