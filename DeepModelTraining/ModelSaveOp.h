#ifndef ModelSaveOp_h
#define ModelSaveOp_h

#include <ECF/ECF.h>
#include "ModelEvalOp.h"
#include <fstream>
#include <windows.h>

class ModelSaveOp : public Operator
{
protected:
  bool m_SaveModel = false;
  std::string m_ModelExportPath;
  ModelEvalOpP m_EvaluationOp;
  SelBestOpP m_SelBestOp;
  std::ofstream m_OutputStream;
  uint m_SaveMilestone;
  double m_TerminatingFitness;
  uint m_TerminatingGeneration;
  // helper function for final evaluation of saved model
  double getAverageTrainingError(IndividualP individual);
public:
  ModelSaveOp() : m_SelBestOp(new SelBestOp) {}
  void registerParameters(StateP state) override;
  bool initialize(StateP state) override;
  bool operate(StateP state) override;

};

typedef boost::shared_ptr<ModelSaveOp> ModelSaveOpP;

#endif
