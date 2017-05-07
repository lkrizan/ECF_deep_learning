#ifndef Layer_h
#define Layer_h

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <common/Shape.h>
#include <common/Factory.h>

namespace NetworkConfiguration {


// helper classes used to unify constructor arguments
struct LayerBaseParams
{
  tensorflow::Scope &scope_;
  const tensorflow::Input &previousLayerOutput_;
  const Shape& previousLayerOutputShape_;
  LayerBaseParams(tensorflow::Scope & scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape) :
    scope_(scope), previousLayerOutput_(previousLayerOutput), previousLayerOutputShape_(previousLayerOutputShape) {};
};

// base shape with one more parameter (for layers which have one additional parameter with shape (FullyConnectedLayer, etc.)
struct LayerShapeL1Params : public LayerBaseParams
{
  const std::vector<int> & paramShapeArgs_;
  LayerShapeL1Params(tensorflow::Scope & scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int>& paramShapeArgs) :
    LayerBaseParams(scope, previousLayerOutput, previousLayerOutputShape), paramShapeArgs_(paramShapeArgs) {};
};

// base shape with 2 more parameters (parameter shape and stride shape: convolution and pooling layers)
struct LayerShapeL2Params : public LayerShapeL1Params
{
  const std::vector<int> & strideShapeArgs_;
  LayerShapeL2Params(tensorflow::Scope & scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const std::vector<int>& paramShapeArgs, const std::vector<int> & strideShapeArgs) :
    strideShapeArgs_(strideShapeArgs), LayerShapeL1Params(scope, previousLayerOutput, previousLayerOutputShape, paramShapeArgs) {};
};


// do not inherit from this class, use this only as an interface
class Layer
{ 
protected:
  // scope for placeholder variables
  tensorflow::Scope &m_Scope;
  // placeholder for output out of the layer
  tensorflow::Output m_Output;
  // placeholder for input in to the layer
  const tensorflow::Input & m_Input;
  Shape m_OutputShape;
  Layer(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput) : m_Scope(scope), m_Input(previousLayerOutput) {};
  Layer(LayerBaseParams & params) : m_Scope(params.scope_), m_Input(params.previousLayerOutput_) {};

public:
  virtual const tensorflow::Output& forward() const { return m_Output; }
  virtual Shape outputShape() const { return m_OutputShape; }
  virtual bool hasParams() const = 0;
  virtual ~Layer() = default;
  // returns gradient over inputs
  virtual tensorflow::Output backwardInputs(const tensorflow::Input & previousInputsGradient) = 0;
};

typedef std::shared_ptr<Layer> LayerP;


// for implementing new layers, inherit from these two classes and make sure you set m_Output and m_OutputShape in class constructor
class NonParameterizedLayer : public Layer
{
protected:
  NonParameterizedLayer(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput) : Layer(scope, previousLayerOutput) {};
  NonParameterizedLayer(LayerBaseParams & params) : Layer(params) {};
public:
  virtual ~NonParameterizedLayer() = default;
  bool hasParams() const override { return false; };
};

typedef std::shared_ptr<NonParameterizedLayer> NonParameterizedLayerP;


class ParameterizedLayer : public Layer
{
protected:
  // actual parameter references
  tensorflow::Output m_Weights;
  tensorflow::Output m_Bias;

  ParameterizedLayer(LayerShapeL1Params & params) : Layer(params) {};
  ParameterizedLayer(tensorflow::Scope & scope, const tensorflow::Input & previousLayerOutput) : Layer(scope, previousLayerOutput) {};

public:
  virtual ~ParameterizedLayer() = default;
  bool hasParams() const override { return true; };
  // returns shapes of all parameters
  virtual std::vector<std::pair<std::string, Shape>> getParamShapes() const = 0;
  const tensorflow::Output & getWeights() const { return m_Weights; }
  const tensorflow::Output & getBias() const { return m_Bias; }
  // returns gradient over weights
  virtual tensorflow::Output backwardWeights(const tensorflow::Input & previousInputsGradient) = 0;
  // returns gradient over bias
  virtual tensorflow::Output backwardBias(const tensorflow::Input & previousInputsGradient) = 0;
};

typedef std::shared_ptr<ParameterizedLayer> ParameterizedLayerP;

typedef std::function<Layer*(LayerBaseParams &)> LayerCreator;
typedef Common::Factory<Layer, std::string, LayerCreator> LayerFactory;

}	// namespace NetworkConfiguration
#endif