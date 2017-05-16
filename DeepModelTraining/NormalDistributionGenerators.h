#ifndef NormalDistributionGenerators_h
#define NormalDistributionGenerators_h

#include "IRNGenerator.h"

template<typename RealType>
class NormalDistributionRNG : public IRNGenerator<RealType>
{
  std::mt19937 m_RNEngine;
  typename std::normal_distribution<RealType> m_Generator;

public:
  NormalDistributionRNG(RealType mean, RealType stddev) : m_Generator(mean, stddev), m_RNEngine(std::random_device()()) {}
  NormalDistributionRNG(RNGBaseParams<RealType> & params) : NormalDistributionRNG(params.firstArgument_, params.secondArgument_) {}
  RealType operator()() override;
};


template<typename RealType>
class TruncatedNormalDistributionRNG : public IRNGenerator<RealType>
{
  RealType m_Mean;
  RealType m_StdDev;
  std::mt19937 m_RNEngine;
  typename std::normal_distribution<RealType> m_Generator;

public:
  TruncatedNormalDistributionRNG(RealType mean, RealType stddev) : m_Generator(mean, stddev), m_RNEngine(std::random_device()())
  {
    m_Mean = mean;
    m_StdDev = stddev;
  }
  TruncatedNormalDistributionRNG(RNGBaseParams<RealType> & params) : TruncatedNormalDistributionRNG(params.firstArgument_, params.secondArgument_) {}
  RealType operator()() override;
};

#endif
