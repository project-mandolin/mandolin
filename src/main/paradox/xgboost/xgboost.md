# XGBoost

XGBoost is a popular machine learning library providing a number of robust, discriminative
classification and regression algorithms.  XGBoost's "flagship" algorithms are based on
the technique of Gradient Boosted Trees. Using XGBoost via Mandolin offers some advantages:

 * Allows for training runs and experiments to be specified via Mandolin configuration files
 * Provides an input format and workflow consistent with MXNet and Mandolin's other ML algorithms
   so that a large suite of algorithms can be easily experimented with on a given dataset
 * Benefits from Mandolin's model selection capabilities. Gradient Boosted Trees and other algorithms
   provided by XGBoost can be difficult to tune. Automated model selection makes this easier
 * Scalability via Spark.

It should be noted that XGBoost that if only XGBoost is needed, it may be preferrable to use
the library directly or via another wrapper (e.g. scikit). 

## Configuration

XGBoost settings are specified via an "xg" block (i.e. namespace) within the configuration file.
Consider the example below:

@@snip [mnist.conf](snippets/splice.1.conf)