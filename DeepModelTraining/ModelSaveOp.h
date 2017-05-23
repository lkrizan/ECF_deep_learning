#ifndef ModelSaveOp_h
#define ModelSaveOp_h

#include <ECF/ECF.h>
#include "ModelEvalOp.h"
#include <fstream>

class ModelSaveOp : public Operator
{
  bool m_SaveModel = false;
  std::string m_ModelExportPath;
  ModelEvalOpP m_EvaluationOp;
  SelBestOpP m_SelBestOp;
  std::ofstream m_OutputStream;

  // helper function for final evaluation of saved model
  double getAverageTrainingError(IndividualP individual);
  // model is saved after the last generation is completed
  uint m_TerminatingGeneration;
public:
  ModelSaveOp() : m_SelBestOp(new SelBestOp) {}
  void registerParameters(StateP state) override;
  bool initialize(StateP state) override;
  bool operate(StateP state) override;

};

typedef boost::shared_ptr<ModelSaveOp> ModelSaveOpP;

#endif
