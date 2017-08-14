Mandolin
========

Mandolin is a tool for developing machine learning-based prediction models. It provides capabilities
such as:

* A rich configuration system for declaratively specifying models
* Fast distributed training using Apache Spark
* Large-scale, robust concurrent model selection/tuning
    * Concurrent evaluation using Apache Spark
    * Adaptive model selection using Bayesian Optimization techniques and Hyperband
    * Declarative specification of the model space
* Deep learning via MXNet
    * Includes support for training with GPUs
    * Specification of complex neural networks via the configuration system
 * Gradient boosted trees via XGBoost
 * Linear models and multi-layered perceptrons with a variety of loss functions
 * Optional and seemless standalone use (without Apache Spark)

Mandolin also contains an API extension based on **spark.ml**.
This allows Mandolin models to be used directly within the MLLib ecosystem, leveraging
other datastructures, tools, and MLLib's rich ML Pipeline framework.

@@@ index

* [Installation](installation/installation.md)
* [Essentials](essentials/index.md)
* [MMLPs](mmlp/mmlp.md)
* [XGBoost](xgboost/xgboost.md)
* [MXNET](mxnet/mxnet.md)
* [Model Selection](model-selection/mselect.md)
* [Spark ML](spark/spark.md)
* [Programmatic Use](api-use/api.md)

@@@