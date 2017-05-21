#include "DLFloatingPointMutUnbound.h"

namespace DLFloatingPoint {

void DLFloatingPointMutUnbound::registerParameters(StateP state)
{
  myGenotype_->registerParameter(state, "mut.unbound", (voidP) new double(0), ECF::DOUBLE);
  myGenotype_->registerParameter(state, "numberGenerator", (voidP) new std::string("NormalDistributionRNG"), ECF::STRING);
  // distribution mean (or something similar)
  myGenotype_->registerParameter(state, "distributionArg1", (voidP) new double(0), ECF::DOUBLE);
  // standard deviation (or something similar, like upper bound)
  myGenotype_->registerParameter(state, "distributionArg2", (voidP) new double(1), ECF::DOUBLE);
}

bool DLFloatingPointMutUnbound::initialize(StateP state)
{
  voidP sptr = myGenotype_->getParameterValue(state, "mut.unbound");
  probability_ = *((double*)sptr.get());
  // set random number generator
  const std::string rngClassName = *static_cast<std::string*>(myGenotype_->getParameterValue(state, "numberGenerator").get());
  double arg1 = *static_cast<double*>(myGenotype_->getParameterValue(state, "distributionArg1").get());
  double arg2 = *static_cast<double*>(myGenotype_->getParameterValue(state, "distributionArg2").get());
  rng_ = RNGFactory<double>::instance().createObject(rngClassName, RNGBaseParams<double>(arg1, arg2));
  return true;
}

bool DLFloatingPointMutUnbound::mutate(GenotypeP gene)
{
  FloatingPoint::FloatingPoint* FP = (FloatingPoint::FloatingPoint*)(gene.get());

  uint dimension = state_->getRandomizer()->getRandomInteger((uint)FP->realValue.size());
  FP->realValue[dimension] += rng_->operator()();

  return true;
}



}
