#ifndef DLFloatingPoint_h
#define DLFloatingPoint_h

#include <ECF/ECF.h>
#include <ECF/floatingpoint/FloatingPoint.h>
#include "DLFloatingPointMutUnbound.h"

namespace DLFloatingPoint {

class DLFloatingPoint : public FloatingPoint::FloatingPoint
{
public:
  DLFloatingPoint() { name_ = "FloatingPoint"; }
  /*
  TODO: in some future update override these methods to be used as a replacement for clunky 
  reinitialization of individuals in ModelEvalOp initialization.

  bool initialize(StateP state) override;
  void registerParameters(StateP state) override;
  */
  DLFloatingPoint* copy()
  {
    DLFloatingPoint *newObject = new DLFloatingPoint(*this);
    return newObject;
  }

  std::vector<MutationOpP> getMutationOp() override
  {
    std::vector<MutationOpP> mut = FloatingPoint::FloatingPoint::getMutationOp();
    mut.push_back(static_cast<MutationOpP>(new DLFloatingPointMutUnbound));
    return mut;
  }

};
}

typedef boost::shared_ptr<DLFloatingPoint::DLFloatingPoint> DLFloatingPointP;
#endif