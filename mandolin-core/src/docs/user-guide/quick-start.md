{%
  title: Quickstart
%}

MNIST Example
-------------

While Mandolin is intended to be used with Apache Spark, all algorithms can be run within
a single JVM process using a separate non-Spark backend. To run a simple example using a linear (logistic
regression) model, enter the directory `examples/mnist`. Assuming the Mandolin jar file
has been built (see above), the following invocation should train a model and evaluate its
progress during training every trainin epoch:

    java -cp ../../mandolin-core/target/scala-2.11/mandolin-core-assembly-0.3_2.11.7.jar \
         org.mitre.mandolin.app.Driver --conf mnist.conf

The evaluation is written in this example to the file `mnist.train.progress` as specified in the
`mnist.conf` configuration file. Note that all configuration file values can be overridden on the
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


MNIST Example Using Spark
------------------------

Training can be very CPU-intensive with neural networks. Mandolin supports distributed
parameter estimation via Apache Spark which can provide faster training for certain
datasets, and model topologies. On each iteration separate models are trained on different compute nodes
over different (possibly overlapping) subsets of the training data. After each epoch (or after
every N epochs) the model parameters (weights) are averaged.  

To run Mandolin on a Spark cluster, ensure that the setting 
`mandolin.without-spark` is set to `false` by either modifying the configuration file or
providing the argument `mandolin.without-spark=false` on the command line.  Then execute

    $SPARK_HOME/bin/spark-submit --class org.mitre.mandolin.app.Driver \
      --master spark://master.url:7077 \
          ../../mandolin-spark/target/scala-2.11/mandolin-spark-assembly-0.3_2.11.7.jar \
      --conf mnist.1h.conf mandolin.without-spark=false


    