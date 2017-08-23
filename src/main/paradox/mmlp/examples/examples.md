# Examples

## MNIST Example


Run the MNIST MMLP example via the following commands, starting from the top-level `mandolin` directory.

    cd examples/mnist
    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin --conf mnist.conf

This invocation's behavior is driven by the contents of the `mnist.conf` file, a portion of which is
below:

@@snip [mnist.conf](snippets/mnist.1.conf)

A key field is the `mode` attribute within the `mandolin` block/namespace. This field, `mandolin.mode` specifies
the overall behavior of the invocation. In this case, the mode is `"train-test"` which specifies that a model
should be trained on the provided training data, specified by the field `mandolin.mmlp.train-file` and
evaluated against the examples in `mandolin.mmlp.test-file`. The field `mandolin.mmlp.eval-freq` specifies
how frequently (in terms of training iterations/epochs) the model should be evaluated against the
test data. The results of this repeated evaluation are presented in the file `mandolin.mmlp.progress-file`.

The various configuration options are discussed in more detail here @ref[config](../configuration.md).

Mandolin encourages the use of configuration files as they provide a written record of experiments and
can be used within version control to allow experiments to be easily repeated and recovered. In some cases,
however it is useful to be able to specify parameters on the command-line. Mandolin configuration
attributes can be overridden on the command line by simply specifying attribute value pairs separated by
the `=` sign. For example, we can change the number of training epochs with the invocation:

    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin --conf mnist.conf \
        mandolin.mmlp.num-epochs=60

Rather than using a single test file to validate the trained model, we may prefer to use cross validation.
In `train-test` mode, if Mandolin is only provided a training file and no testing file, the assumed
intent is to perform cross validation on the training file. We can achieve this by simply setting
the test file to `null` as follows:

    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin --conf mnist.conf \
        mandolin.mmlp.test-file=null

When performing cross validation in `train-test` mode, Mandolin will not perform an evaluation
after each epoch and instead simply evaluate the final models. The evaluation metric results
on each fold will be aggregated with the results placed in the contents of the file
`mandolin.mmlp.progress-file`.

Mandolin supports a variety of model topologies and loss functions which can be specified easily
using the configuration system. There are also various methods for optimizing loss functions. The relevant
configuration options within `mandolin.conf` are:

@@snip [mnist.conf](snippets/mnist.2.conf)

The options within the `mandolin.mmlp.optimizer` block indicate which optimization algorithm should be used
and any relevant parameters, in this case Adagrad is used (via `mandolin.mmlp.optimizer.method=adagrad`) and
the initial learning rate is set to 0.1.

The model topology is specified within the `mandolin.mmlp.specification` attribute which takes a list
of layers. In this case, the network is linear having just an input layer and a "softmax" output layer.

More details follow in the configuration section.

Moving on to a more complex example, let's use the same data, but add a hidden layer to create
a multi-layered perceptron. Specifically, let's add a single hidden layer with 500 units/dimensions
with the rectified linear activation function.  

This model is specified in the configuration file `examples/mnist/mnist.1hl.conf`. The relevant
portions are:

@@snip [mnist.1hl.conf](snippets/mnist.1hl.1.conf)

The model can be invoked as:

    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin \
         --conf mnist.1hl.conf

Training this model requires markedly longer. The results are better, however, as this
model has more capacity and can capture non-linear interactions. In fact, adding two more layers
of comparable complexity allows the model to reach error rates of less than 1% when using
the full MNIST training set.

## Web URL
The Web URL dataset contains about 2.4M instances of sparsely-populated feature vectors. To benefit from the
sparse vector optimizations, the `mandolin.mmlp.ensure-sparse` parameter is set to `true`.  Also, the layer type of
the input layer in the mmlp specification has changed from `Input` to `InputSparse`.

@@snip [url.conf](snippets/url.conf)

The Web URL training example can be invoked as:

    cd examples/weburl
    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin --conf url.conf