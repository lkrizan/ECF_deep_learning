#ifndef AlgMicrocanonicalAnnealing_h
#define AlgMicrocanonicalAnnealing_h

#include <ECF/ECF.h>
#include <ECF/Algorithm.h>

class MicrocanonicalAnnealing : public Algorithm
{
public:
  MicrocanonicalAnnealing() { name_ = "MicrocanonicalAnnealing"; }

  void registerParameters(StateP state) override;
  bool initialize(StateP state) override;
  bool advanceGeneration(StateP state, DemeP deme) override;

protected:
  double demon_;
  double upperBound_;
};

typedef boost::shared_ptr<MicrocanonicalAnnealing> MicrocanonicalAnnealingP;
#endif