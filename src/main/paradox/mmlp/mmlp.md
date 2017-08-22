# MMLPs


Mandolin supports *built-in* iterative gradient-based machine learning algorithms that do
not rely on third party libraries and run entirely within the JVM (implemented in
Scala). These algorithms are robust, do not rely on native code, and offer efficient
efficient training for high-dimensional, sparse, linear models such as multinomial
logistic regression and linear regression and variations with different loss functions.
These models can incorporate fully connected hidden layers resulting
in multi-layered perceptron models, yet lack efficient native or GPU-accelerated
linear algebra routines. Accordingly, these models can be relatively slow to train. In addition, modern
deep learning libraries (such as MXNet or Tensorflow)
include a much more extensive array of options for multi-layered
perceptrons as well as more complex neural network architectures.

## MMLP Configuration

These models are specified in the configuration file within the namespace `mandolin.mmlp`.
Important configuration options/settings for these models include:

|  Property Name                             |  Meaning              |
| -----------------------------------------: | --------------------- |
|``mandolin.mmlp.train-file``             |Training file in the standard Mandolin classification input format. |
|``mandolin.mmlp.model-file``             |String path specifying where the trained model should be written to. The model is represented as Kryo serialized objects including the model weights and any auxiliary information (e.g. feature alphabet) required to use the model for inference on new data. |
|``mandolin.mmlp.test-file``              |A development test file to use for evaluation during/after the training process. |
|``mandolin.mmlp.num-epochs``             |Number of training epochs.|
|``mandolin.mmlp.progress-file``          |File path where output showing training progress measured by training loss as well as accuracy, area under the ROC curve and other metrics against the test data provided by |
|``mandolin.mmlp.scale-inputs``           |Scale inputs so that they have a range in `[0,1]`|
|``mandolin.mmlp.specification``          |A JSON-syntax list/array specification for the multi-layered perceptron topology, including activation functions, dimensions of hidden layers and the output layer type and loss function|

## Layer Specification

MMLPs are specified with a simple array/list that indicates the type of each layer
in a multi-layered perceptron.  The first layer is always an input layer with one of
two types: ``Input`` or ``InputSparse``. The latter assumes a sparse representation
and will use various optimizations for efficient backpropagation from a dense hidden layer
to a sparse input layer. The final layer specifies an output range, dimensionality as well
as the corresponding loss function (explicit or implied). For classification tasks, the output
layer is typically ``SoftMax``. Hidden layers are of type ``Relu``, ``TanH`` or ``Logistic``
corresponding to the type of activation function for the layer and must also specify the
number of dimensions for the layer.  Finally, each layer, except the input layer may
include L1, L2 or max-norm regluarization as well as drop-out.


## Additional properties

Some additional options are detailed below:

|  Property Name                             |  Meaning              |
| -----------------------------------------: | --------------------- |
|<nobr>`mandolin.mmlp.dense-vector-size`</nobr> | An ``integer`` that denotes the number of dimensions in the inputs.  This is generally used only when the inputs are dense.  If set to 0 or left unset the input reader will create a symbol table mapping feature names to integers. |
|<nobr>`mandolin.mmlp.ensure-sparse`</nobr>     | If set to ``true``, this will force Mandolin to try to use a sparse vector representation when reducing model weights or gradients (with batch training) |
|``mandolin.mmlp.max-features-mi``        |``Integer`` specifying the number of features to select independently using Mutual Information prior to training. |
|``mandolin.mmlp.eval-freq``              |Frequency of training progress evaluation which is rendered to the output specified by ``mandolin.mmlp.progress-file``|
|``mandolin.mmlp.test-partitions``        |Specifies the number of data partitions for the *test* data specified by ``mandolin.mmlp.test-file``. This should usually be set to 3-4 times the number of total cores on the compute cluster as typical for Spark jobs.|
|``mandolin.mmlp.oversample``             |This is a ``Double`` that is a coefficient determining how much data should be (re)sampled  from the entire dataset for each partition/node during online training. If this is non-positive, the dataset is not resampled and 1/k of the data (where k is the number of compute nodes) is sharded to each compute node. If set to a positive value, a, then a/k of the data is sampled for each data partition **on each epoch**.  See user guide for more details.|
|``mandolin.mmlp.print-feature-file``     |A file path that will receive a list of all the features used/selected for training with a given input file.|


@@@ include

 * [Examples](examples/examples.md)

@@@

