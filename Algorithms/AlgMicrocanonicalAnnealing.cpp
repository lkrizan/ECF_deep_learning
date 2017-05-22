#include "AlgMicrocanonicalAnnealing.h"

void MicrocanonicalAnnealing::registerParameters(StateP state)
{
  registerParameter(state, "demon", (voidP) new double(2), ECF::DOUBLE);
}

bool MicrocanonicalAnnealing::initialize(StateP state)
{
  demon_ = *static_cast<double*>(getParameterValue(state, "demon").get());
  upperBound_ = demon_;
  return true;
}

bool MicrocanonicalAnnealing::advanceGeneration(StateP state, DemeP deme)
{
  IndividualP ind = deme->at(0);
  IndividualP mutant = static_cast<IndividualP>(ind->copy());
  mutate(mutant);
  evaluate(mutant);
  double energyDiff = mutant->fitness->getValue() - ind->fitness->getValue();
  if (energyDiff <= demon_ )
  {
    deme->at(0) = mutant;
    demon_ -= energyDiff;
    if(demon_ > upperBound_) demon_ = upperBound_;
  }
  return true;
}