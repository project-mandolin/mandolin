# MXNet

Mandolin provides a wrapper around MXNet that makes it easier to train certain types
of MXNet models. Currently, Mandolin provides support for training convolutional neural
networks and other feed-forward models that can be specified via symbolically(declaratively)
Network architectures that are specified  procedurally, such as recurrent neural networks (RRNs),
are not yet supported. Besides making the model specification simpler via a configuration
file, Mandolin provides model selection over many of the hyper-parameters available for MXNet
neural networks.

## Configuration 

|  Property Name                             |  Meaning              |
| -----------------------------------------: | --------------------- |
|<nobr>``mandolin.mx.model-file``</nobr>     |String path specifying where the trained model should be written to. The model is represented as Kryo serialized objects that include the necessary configuration settings for
extracting and representing features as well as the MXNet model weights. |
|``mandolin.mx.num-epochs``             |Number of training epochs.|
|<nobr>``mandolin.mx.progress-file``</nobr>  |File path where output showing training progress measured by training loss as well as accuracy, area under the ROC curve and other metrics against the test data provided by |
|``mandolin.mx.test-file``              |A development test file to use for evaluation during/after the training process. |
|``mandolin.mx.train-file``             |Training file in the standard Mandolin classification input format. |
|``mandolin.mx.eval-freq``              |Frequency of training progress evaluation which is rendered to the output specified by ``mandolin.mx.progress-file``|
