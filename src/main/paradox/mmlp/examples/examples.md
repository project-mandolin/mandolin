# Examples

## MNIST Example


Run the MNIST MMLP example via the following commands, starting from the top-level `mandolin` directory.

    cd examples/mnist
    java -cp ../../dist/mandolin-core-0.3.5.jar org.mitre.mandolin.app.Mandolin --conf mnist.conf

This invocation's behavior is driven by the contents of the `mnist.conf` file. In this instance, the behavior
involves training model on the providing training data and evaluating the results (after each epoch) on
a single set of validation/test data. The results are written to the file `mnist.train.progress`.

Below is a portion of the `mnist.conf` file specifying the behavior requested above:

@@snip [mnist.conf](snippets/mnist.1.conf)

Note that all configuration file values can be overridden on the
command line.  So to write the training progress to a different file name, one could use the 
invocation:

    java -cp ../../mandolin-core/target/scala-2.11/mandolin-core-assembly-0.3_2.11.7.jar \
        org.mitre.mandolin.app.Driver --conf mnist.conf \
        mandolin.app.trainer.progress-file=$PWD/another.output.file

Mandolin supports a variety of model topologies and loss functions which can be specified easily
using the configuration system. More details follow in the configuration section, but for now,
let's train a DNN model using a single hidden layer of 1000 Rectified Linear Units with 50% dropout.
This model is specified in the configuration file `examples/mnist/mnist.1hl.conf` and can be
invoked as:

    java -cp ../../mandolin-core/target/scala-2.11/mandolin-core-assembly-0.3_2.11.7.jar \
         org.mitre.mandolin.app.Driver --conf mnist.conf \
         --conf mnist.1h.conf 

Note that the training of this model will take some time to finish. The configuration file specifies
that 8 CPU threads should be used. If you have a more powerful machine, you can speed up the training
time by adding the option `mandolin.threads=N` to the command-line
invocation where `N` refers to the number of threads to utilize.

Examination of the output file `mnist.train.1hl.progress` makes 
apparent the advantage of using a more complex, non-linear model on this dataset. The test accuracies are 
much improved over the linear model.

## Web URL