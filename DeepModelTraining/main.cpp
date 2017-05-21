#include <ECF/ECF.h>
#include "ModelEvalOp.h"
#include <Genotype/DLFloatingPoint.h>
#include <Algorithms/AlgBackpropagation.h>


int main(int argc, char **argv)
{
  StateP state(new State);

  DLFloatingPointP gen(new DLFloatingPoint::DLFloatingPoint());
  BackpropagationP alg(new Backpropagation);
  state->addGenotype(gen);
  state->addAlgorithm(alg);

  // set the evaluation operator
  ModelEvalOp evalOp;
  state->setEvalOp(&evalOp);

  state->initialize(argc, argv);
  state->run();
  return 0;
}