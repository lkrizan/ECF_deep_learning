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

struct LayerShapeParams : LayerBaseParams
{
  const Shape & paramShape_;
  LayerShapeParams(tensorflow::Scope & scope, const tensorflow::Input &previousLayerOutput, const Shape& previousLayerOutputShape, const Shape& paramShape) :
    LayerBaseParams(scope, previousLayerOutput, previousLayerOutputShape), paramShape_(paramShape) {};
};


// do not inherit from this class, use this only as an interface
class Layer
{ 
protected:
  // scope for placeholder variables
  tensorflow::Scope &m_Scope;
  // placeholder for output out of the layer
  tensorflow::Output m_Output;
  Shape m_OutputShape;
  Layer(tensorflow::Scope & scope) : m_Scope(scope) {};
  Layer(LayerBaseParams & params) : m_Scope(params.scope_) {};

public:
  virtual const tensorflow::Output& forward() const { return m_Output; }
  virtual Shape outputShape() const { return m_OutputShape; }
  virtual bool hasParams() const = 0;
  virtual ~Layer() = default;
};

typedef std::shared_ptr<Layer> LayerP;


// for implementing new layers, inherit from these two classes and make sure you set m_Output and m_OutputShape in class constructor
class NonParameterizedLayer : public Layer
{
protected:
  NonParameterizedLayer(tensorflow::Scope & scope) : Layer(scope) {};
  NonParameterizedLayer(LayerBaseParams & params) : Layer(params) {};
public:
  virtual ~NonParameterizedLayer() = default;
  bool hasParams() const override { return false; };
};

typedef std::shared_ptr<NonParameterizedLayer> NonParameterizedLayerP;


class ParameterizedLayer : public Layer
{
protected:
  ParameterizedLayer(LayerShapeParams & params) : Layer(params) {};
  ParameterizedLayer(tensorflow::Scope & scope) : Layer(scope) {};
public:
  virtual ~ParameterizedLayer() = default;
  bool hasParams() const override { return true; };
  // returns shapes of all parameters
  virtual std::vector<std::pair<std::string, Shape>> getParamShapes() const = 0;
};

typedef std::shared_ptr<ParameterizedLayer> ParameterizedLayerP;

typedef std::function<Layer*(LayerBaseParams &)> LayerCreator;
typedef Common::Factory<Layer, std::string, LayerCreator> LayerFactory;

}	// namespace NetworkConfiguration
#endif