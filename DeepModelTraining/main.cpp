#include <ECF/ECF.h>
#include "ModelEvalOp.h"
#include <Algorithms/AlgBackpropagation.h>


int main(int argc, char **argv)
{
  StateP state(new State);

  BackpropagationP alg(new Backpropagation);
  state->addAlgorithm(alg);

  // set the evaluation operator
  ModelEvalOp evalOp;
  state->setEvalOp(&evalOp);

  state->initialize(argc, argv);
  state->run();
  return 0;
}