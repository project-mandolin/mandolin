Configuration
=============

Mandolin provides various ways to provide configuraiton properties that specifiy system
behavior. Primarily this is done using a configuration file.

|  Property Name                             |  Meaning              |
| -----------------------------------------: | --------------------- |
| ``mandolin.partitions``                    | Number of partitions/nodes to use for online distributed optimization. |
| ``mandolin.name``                          | Provide the name of the application.              |
| ``mandolin.data``                           |Specify a root directory for data to be used by the application |
|<nobr>`mandolin.trainer.dense-vector-size`</nobr> | An ``integer`` that denotes the number of dimensions in the inputs.  This is generally used only when the inputs are dense.  If set to 0 or left unset the input reader will create a symbol table mapping feature names to integers. |
|<nobr>`mandolin.trainer.ensure-sparse`</nobr>     | If set to ``true``, this will force Mandolin to try to use a sparse vector representation when reducing model weights or gradients (with batch training) |
|``mandolin.trainer.model-file``             |String path specifying where the trained model should be written to. The model is represented as Kryo serialized objects including the model weights and any auxiliary information (e.g. feature alphabet) required to use the model for inference on new data. |
|``mandolin.trainer.num-epochs``             |Number of training epochs.|
|``mandolin.trainer.max-features-mi``        |``Integer`` specifying the number of features to select independently using Mutual Information prior to training. |
|``mandolin.trainer.progress-file``          |File path where output showing training progress measured by training loss as well as accuracy, area under the ROC curve and other metrics against the test data provided by |
|``mandolin.trainer.test-file``              |A development test file to use for evaluation during/after the training process. |
|``mandolin.trainer.train-file``             |Training file in the standard Mandolin classification input format. |
|``mandolin.trainer.eval-freq``              |Frequency of training progress evaluation which is rendered to the output specified by ``mandolin.trainer.progress-file``|
|``mandolin.trainer.test-partitions``        |Specifies the number of data partitions for the *test* data specified by ``mandolin.trainer.test-file``. This should usually be set to 3-4 times the number of total cores on the compute cluster as typical for Spark jobs.|
|``mandolin.trainer.oversample``             |This is a ``Double`` that is a coefficient determining how much data should be (re)sampled  from the entire dataset for each partition/node during online training. If this is non-positive, the dataset is not resampled and 1/k of the data (where k is the number of compute nodes) is sharded to each compute node. If set to a positive value, a, then a/k of the data is sampled for each data partition **on each epoch**.  See user guide for more details.|
|``mandolin.trainer.print-feature-file``     |A file path that will receive a list of all the features used/selected for training with a given input file.|
|``mandolin.trainer.scale-inputs``           |Scale inputs so that they have a range in `[0,1]`|
|``mandolin.threads``                        |Numberof threads to use for asynchronous gradient updates on each node; should generally match the number of cores available on each compute node. |
|``mandolin.synchronous``                    |Default ``false``. Uses asynchronous stochastic gradient updates; requires that gradient computation is thread safe. If gradient computation is not thread safe (reentrant) then this should be set to ``true`` |
|``mandolin.skip-probability``               |Default ``0.0``. Specify a probability of skipping a training instance during online updating. Can be used for fast undersampling and injection of randomness into estimation without a need to reshuffle data across teh cluster. |
|``mandolin.mini-batch-size``                |Number of data points/examples within which to compute local gradients. Default `1` |
