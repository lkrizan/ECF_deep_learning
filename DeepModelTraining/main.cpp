#include <ECF/ECF.h>
#include "ModelEvalOp.h"


int main(int argc, char **argv)
{
  StateP state(new State);
  // set the evaluation operator
  ModelEvalOp evalOp;
  state->setEvalOp(&evalOp);

  state->initialize(argc, argv);
  state->run();
    return 0;
}