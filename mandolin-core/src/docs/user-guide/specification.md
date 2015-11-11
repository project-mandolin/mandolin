{%
  title: Model Specification
%}

GLP models are specified through a simple JSON-based syntax that provides
an ordered list of objects describing each layer in a feed-forward artificial
neural network. 

## Syntax

Each layer specification must a layer type denoted with an `ltype` attribute.
Additional options for the layer for regularization are then provided as part of the layer
object, all denoted with JSON syntax.

Example:

    mandolin.trainer.specification = 
      [{"ltype":"Input", "dropout-ratio":0.2},
       {"ltype":"Relu","dim":50,"l1-pen":0.00001},
       {"ltype":"SoftMax","l1-pen":0.001}]

The above example specifies three layers - i.e. an input layer, an output layer
of type *SoftMax* and a single hidden layer using Rectified Linear activations. 
Additional options for each layer are specified through attribute value pairs using
JSON syntax. For example, the above specification indicates 20% dropout (i.e. masking
noise) for the input layer, and different L1 regularization penalty terms for the
single hidden layer and output layer.

## Input Layers

Input layer types include **Input** and **InputSparse**, the latter denoting that 
the input is sparse and sparse vector and matrix representations should be used
for the inputs and weights between the first and second layers. 

### Options

The only additional specification for an input layer is *dropout-ratio*. This adds
masking noise to non-zero features according to the ratio/percentage specified.

## Hidden Layers

Layer types include **Logistic**, **TanH**, **Relu** (Rectified linear units) and **Linear**. 
These functions correspond to those described in the literature. 
In addition to the type, for each hidden layer the dimensionality (i.e. number of nodes/neurons)
must be specified.  This is done using the "dim" attribute. Each hidden layer can optionally
be regularized in a few different ways:

### Options

**dropout-ratio**: Just as with input layers, the activations from previous layers can
be "dropped out" randomly.  See the paper on this: *Dropout: A Simple Way to Prevent 
Neural Networks from Overfitting*, Hinton et al..

**l1-pen**: This option specifies an L1 regularizer on the weights for this layer. Larger
penalty terms introduce stronger regularization

**l2-pen**: This option specifies an L2 regularizer on the weights.  Currently only either
L1 or L2 regularization is supported.  If both penalty terms are present, L1 regularization
takes precedence and the L2 penalty will be ignored.

**max-norm**: Max-norm enforces that the L2 norm of the weights is less than the specified value;
thus smaller values here provide stronger regularization.  This type of regularization is 
especially useful in conjunction with Relu layers as the weights can receive large values as 
the gradients don't "vanish" as with sigmoid-type activations.

## Output Layers

Currently, Mandolin is focused on classification algorithms rather than 
regression models. Output layers are thus geared towards multiclass classification.
The primary (default) output layer is the *SoftMax* layer, however other
output layer types that provide different types of cost/loss functions.

**SoftMax**: Softmax layer implements a cross entropy loss where the linear output activations
are transformed with the softmax function.

**Hinge**: This is an "L1" multiclass hinge loss as proposed by Crammer and Singer. 

**ModHuber**: This is a modified Huber loss geared for classification problems, c.f.
*Solving large scale linear prediction problems using stochastic gradient descent algorithms. ICML.*.

**Ramp**: This is a nonconvex loss function. It has a hinge loss but flattens out so that examples 
"strongly" misclassified by the model do not contribute to the loss.  This effectively *ignores*
hard-to-classify examples and can be useful in situations where significant lable noise is present.
See *Trading Convexity for Scalability*, Collobert, Sinz, Weston, Bottou. (ICML 2006).

**Translog**: This is a smooth version of the ramp loss, non-convex

**T-Logistic**: Another smooth non-convex loss suitable for problems with label noise, cf. 
*t-Logistic Regression*, Ding and Vishwanathan. 

### Options

Each output layer can take l1, l2 or max-norm regularization options. 