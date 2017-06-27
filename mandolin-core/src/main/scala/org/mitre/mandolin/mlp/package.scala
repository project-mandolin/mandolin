package org.mitre.mandolin
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

/**
 * == Deep and Linear Learning ==
 * Package provides classes for Deep Neural Networks. Networks without
 * any hidden layers correspond to linear models. Model inputs can be
 * sparse or dense with appropriate optimizations and datastructures
 * for each. Various SGD optimization routines are available such
 * as plain SGD, SGD with momentum, Nesterov accelerated SGD, AdaGrad,
 * RMSProp and AdaDelta. Regularization including L1 and L2 regularization,
 * MaxNorm regularization and Dropout Regularization are available.
 * Network configurations are specified within application configuration
 * files.  See the User Guide for more details.
 * 
 * The mlp includes a
 * variety of loss functions available for all model types (linear and
 * non-linear/deep models) including hinge, log (i.e. SoftMax with cross-entropy) and modified
 * Huber loss as well as non-convex loss functions such as the ramp loss
 * and the T-logistic loss. These loss functions are geared towards categorical 
 * classification (all loss functions work with non-binary, categorical dependent
 * variables) rather than regression, currently.
 */
package object mlp {

}
