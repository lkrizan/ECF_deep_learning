#ifndef FloatingPointMutUnbound_h
#define FloatingPointMutUnbound_h

#include <ECF/ECF.h>
#include <DeepModelTraining/IRNGenerator.h>

namespace DLFloatingPoint {

class DLFloatingPointMutUnbound : public MutationOp
{
protected:
  RNGeneratorP<double> rng_;
public:
  void registerParameters(StateP state);
  bool initialize(StateP state);
  bool mutate(GenotypeP gene);

};

typedef boost::shared_ptr<DLFloatingPointMutUnbound> DLFloatingPointMutUnboundP;
}

#endif