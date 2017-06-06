{%
  title: Configuration
%}

Mandolin provides various ways to provide configuration properties that specifiy system
behavior. Primarily, this is done using a configuration ''file''.

+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|**Property Name**                           |**Type**                                          |**Description**                                   |**Legal Values**                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.partitions``                     |integer                                           |Number of partitions to use for online            | > 0 - generally, should be equal to the number   |
|                                            |                                                  |distributed optimization.  Valid for              | of executors in the Spark cluster                |
|                                            |                                                  |distributed **TODO add doc for invocation modes** |                                                  |
|                                            |                                                  |invocation only.                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.name``                           |string                                            |Provide the name of the application.              |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.mode``                           |string                                            |Controls model training and validation modes.     |``train``, ``decode``, ``train-test``,            |
|                                            |                                                  |In ``train`` mode , a model is trained on the     |``train-decode``, **TODO add xvalidation**        |
|                                            |                                                  |supplied data and written to the specified file   |                                                  | 
|                                            |                                                  |In ``decode`` mode (not valid for model selection |                                                  |
|                                            |                                                  |invocation), an existing model is applied to the  |                                                  |
|                                            |                                                  |input data and the resulting predictions are      |                                                  |
|                                            |                                                  |written to a file. In ``train-test`` mode, a model|                                                  |
|                                            |                                                  |is trained on the training data and validation is |                                                  |
|                                            |                                                  |performed on separate test data. In               |                                                  |
|                                            |                                                  |``train-decode`` mode (not valid for model        |                                                  |
|                                            |                                                  |selection invocation), a model is trained to      |                                                  |
|                                            |                                                  |completion on the training data, then applied to  |                                                  |
|                                            |                                                  |the test data to get predictions.                 |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.data``                           |string                                            |Specify a root directory for data to be used by   |                                                  |
|                                            |                                                  |the application.  Path must be accessible from all|                                                  |
|                                            |                                                  |nodes in distributed invocation.                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|                                                                                       **TRAINER PROPERTIES**                                                                                        |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.train-file``             |string                                            |Path to training file in the standard Mandolin    |                                                  |
|                                            |                                                  |classification input format.                      |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.test-file``              |string                                            |Path to validation file for ``train-test`` mode   |                                                  |
|                                            |                                                  |                                                  |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.detail-file``            |string                                            |File to log training progress; used instead of    |                                                  |
|                                            |                                                  |progress-file when a test-file is not provided.???|                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.dense-vector-size``      |integer                                           |Denotes the number of dimensions in the inputs.   |                                                  |
|                                            |                                                  |This is generally used only when the inputs are   |                                                  |
|                                            |                                                  |dense.  If set to 0 or left unset, the input      |                                                  |
|                                            |                                                  |reader will create a symbol table mapping feature |                                                  |
|                                            |                                                  |names to integers.                                |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.ensure-sparse``          |boolean                                           |Force Mandolin to try to use a sparse vector    rue/false                                        |
|                                            |                                                  |representation when reducing model weights or     |                                                  |
|                                            |                                                  |gradients (with batch training)                   |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.model-file``             |string                                            |Path specifying where the trained model           |                                                  |
|                                            |                                                  |should be written to. The model is represented as |                                                  |
|                                            |                                                  |Kryo serialized objects including the model       |                                                  |
|                                            |                                                  |weights and any auxiliary information             |                                                  |
|                                            |                                                  |(e.g. feature alphabet) required to use the model |                                                  |
|                                            |                                                  |for inference on new data.                        |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.num-epochs``             |integer                                           |Number of training epochs. No default - this must | > 0                                              |
|                                            |                                                  |be set by the application.                        |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.num-subepochs``          |integer                                           |Number passes over data on each partition during  | > 0                                              |
|                                            |                                                  |each map-reduce epoch. Default: 1                 |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.max-features-mi``        |integer                                           |``Integer`` specifying the number of features to  |                                                  |
|                                            |                                                  |select independently using Mutual Information     |                                                  |
|                                            |                                                  |prior to training.                                |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.progress-file``          |string                                            |File path where output showing training progress  |                                                  |
|                                            |                                                  |measured by training loss as well as accuracy,    |                                                  |
|                                            |                                                  |area under the ROC curve and other metrics against|                                                  |
|                                            |                                                  |the test data provided by                         |                                                  |
|                                            |                                                  |``mandolin.trainer.test-file``                    |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|                                            |                                                  |                                                  |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|                                            |                                                  |                                                  |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.eval-freq``              |integer                                           |Frequency of training progress evaluation which is|                                                  |
|                                            |                                                  |rendered to the output specified by               |                                                  |
|                                            |                                                  |``mandolin.trainer.progress-file``                |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.test-partitions``        |integer                                           |Specifies the number of data partitions for the   |                                                  |
|                                            |                                                  |*test* data specified by                          |                                                  |
|                                            |                                                  |``mandolin.trainer.test-file``. This should       |                                                  |
|                                            |                                                  |usually be set to 3-4 times the number of total   |                                                  |
|                                            |                                                  |cores on the compute cluster as typical for Spark |                                                  |
|                                            |                                                  |jobs.                                             |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.oversample``             |real                                              |This is a ``Double`` that is a coefficient        |                                                  |
|                                            |                                                  |determining how much data should be (re)sampled   |                                                  |
|                                            |                                                  |from the entire dataset for each partition/node   |                                                  |
|                                            |                                                  |during online training. If this is non-positive,  |                                                  |
|                                            |                                                  |the dataset is not resampled and 1/k of the data  |                                                  |
|                                            |                                                  |(where k is the number of compute nodes) is       |                                                  |
|                                            |                                                  |sharded to each compute node. If set to a positive|                                                  |
|                                            |                                                  |value, a, then a/k of the data is sampled for each|                                                  |
|                                            |                                                  |data partition **on each epoch**.  See user guide |                                                  |
|                                            |                                                  |for more details.                                 |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.print-feature-file``     |string                                            |A file path that will receive a list of all the   |                                                  |
|                                            |                                                  |features used/selected for training with a given  |                                                  |
|                                            |                                                  |input file.                                       |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.scale-inputs``           |boolean                                           |Scale inputs so that they have a range in         |                                                  |
|                                            |                                                  |``[0,1]``                                         |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.trainer.threads``                |integer                                           |Number of threads to use for asynchronous gradient|                                                  |
|                                            |                                                  |updates on each node; should generally match the  |                                                  |
|                                            |                                                  |number of cores available on each compute node.   |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.synchronous``                    |boolean                                           |Default ``false``. Uses asynchronous stochastic   |                                                  |
|                                            |                                                  |gradient updates; requires that gradient          |                                                  |
|                                            |                                                  |computation is thread safe. If gradient           |                                                  |
|                                            |                                                  |computation is not thread safe (reentrant) then   |                                                  |
|                                            |                                                  |this should be set to ``true``                    |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.skip-probability``               |real                                              |Default ``0.0``. Specify a probability of skipping|                                                  |
|                                            |                                                  |a training instance during online updating. Can be|                                                  |
|                                            |                                                  |used for fast undersampling and injection of      |                                                  |
|                                            |                                                  |randomness into estimation without a need to      |                                                  |
|                                            |                                                  |reshuffle data across the cluster.                |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|``mandolin.mini-batch-size``                |integer                                           |Number of data points/examples within which to    |                                                  |
|                                            |                                                  |compute local gradients. Default ``1``            |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|                                            |                                                  |                                                  |                                                  |
|                                            |                                                  |                                                  |                                                  |
|                                            |                                                  |                                                  |                                                  |
+--------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+


