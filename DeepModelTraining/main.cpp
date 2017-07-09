#include <ECF/ECF.h>
#include "ModelEvalOp.h"
#include "ModelSaveOp.h"
#include <Genotype/DLFloatingPoint.h>
#include <Algorithms/AlgBackpropagation.h>
#include <Algorithms/AlgMicrocanonicalAnnealing.h>


int main(int argc, char **argv)
{
  StateP state(new State);

  DLFloatingPointP gen(new DLFloatingPoint::DLFloatingPoint());
  BackpropagationP alg1(new Backpropagation);
  MicrocanonicalAnnealingP alg2(new MicrocanonicalAnnealing);
  ModelSaveOpP op1(new ModelSaveOp);
  state->addOperator(op1);
  state->addGenotype(gen);
  state->addAlgorithm(alg1);
  state->addAlgorithm(alg2);

  // set the evaluation operator
  ModelEvalOp evalOp;
  state->setEvalOp(&evalOp);

  state->initialize(argc, argv);
  state->run();
  return 0;
}