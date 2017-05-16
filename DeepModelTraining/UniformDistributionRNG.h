#ifndef UniformDistributionRNG_h
#define UniformDistributionRNG_h

#include "IRNGenerator.h"

template<typename RealType>
class UniformDistributionRNG : public IRNGenerator<RealType>
{
  std::mt19937 m_RNEngine;
  typename std::uniform_real_distribution<RealType> m_Generator;

public:
  UniformDistributionRNG(RealType lbound, RealType ubound) : m_Generator(lbound, ubound), m_RNEngine(std::random_device()()) {}
  UniformDistributionRNG(RNGBaseParams<RealType> & params) : UniformDistributionRNG(params.firstArgument_, params.secondArgument_) {}
  RealType operator()() override;
};

#endif
