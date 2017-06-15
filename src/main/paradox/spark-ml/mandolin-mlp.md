# Multi-Layered Perceptron

Mandolin provides a rich means to specify a wide variety of models that broadly fall in the class
of mutli-layer perceptrons (or feedforward, fully connected neural networks).
Mandolin provides a very general form of MLPs, including a variety of loss functions,
activation functions and forms of regularization.  
Linear models are simply a special case (without any 
hidden nodes), but Mandolin includes sparse data representations and optimizations to 
make learning very large linear models efficient.

Mandolin's configuration system lets users easily fine-tune the configuration of MLPs 
as well as to specify various details for how to optimize the loss function of the MLP 
within a distributed computing environment. The configuration system allows this to
be done without writing code.  This may be useful for experimentation with static
datasets, but in some cases, a MLP might be the basis of an analytic embedded within
a larger workflow.  Mandolin follows the approach of MLLib by providing an API
that allows for the development of arbitrary workflows; more specifically, it allows
for 

*  Datasets to be provided to Mandolin as Spark 
[DataFrames](http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframes), 
*  Construction of distributed stochastic gradient descent updaters (e.g. AdaGrad)
*  Specification of MLPs (including loss functions and network topology)
*  Training MLPs - i.e. estimating model parameters
*  Using trained MLPs to make predictions

The API is in its early stages and provides basic interoperability with Apache Spark's
MLLib **spark.ml** API.

Building upon the MNIST example earlier, the following steps outline how to load
in a dataset as a Spark DataFrame, train a model and evaluate it against
a held out test set.  The API here is found in the package `org.mitre.mandolin.ml`
and extends the `spark.ml` API.  It requires data in a spark DataFrame conforming
to the spark.ml schema for representing datasets for use within MLlib. 

To run this example, start-up a Spark Shell instance (or embedd within an application).
Ensure that Kryo serialization is used and any other Spark configuration options
relevant to your cluster/environment.  For example, to start the shell with a "local"
master, from the mandolin/examples/mnist directory execute:

    $SPARK_HOME/bin/spark-shell \
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
    --conf spark.kryo.registrator=org.mitre.mandolin.config.MandolinKryoRegistrator \
    --master "local[16]" \
    --jars ../../scala-2.11.7/mandolin-assembly-0.3.0-SNAPSHOT_2.11.7.jar
    

Once the shell has been started, the following code can be executed to load data, train a
model, use it for prediction and evaluate:

@@snip [TrainPredict.scala](snippets/TrainPredict.scala)

See the [spark.ml](http://spark.apache.org/docs/latest/ml-guide.html) programming guide
for additional examples of evaluating models, using models in more complex pipelines,
use within ensembles, cross validation and model selection.

### Additional Mandolin Shell Examples

The following code gists show how some of the same can be carried
out without the use of `spark.ml` using Mandolin's own classifier
evaluation routines.  

The following example assumes Mandolin has been launched as part of the
Spark Shell:

    import org.mitre.mandolin.glp._
    val glp = new GlpModel
    val df = glp.readAsDataFrame(sqlContext, sc, "mnist.10k", 784, 10)
    val data = df.randomSplit(Array(0.8,0.2))
    val trData = data(0)
    val tstData = data(1)
    val layerSpec = IndexedSeq(LType(InputLType), LType(SoftMaxLType)) // linear model
    val model = glp.estimate(sc, trData, layerSpec)

    // evaluate model and return a confusion object with eval metrics
    val confusion = glp.evaluate(sc, model, tstData)
    print("Area under ROC: " + confusion.getTotalAreaUnderROC())
    
    // pretty print the confusion matrix when applying the learned model to the test data
    confusion.getMatrix.prettyPrint
    
If we now want to train this with a hidden layer, using 300 nodes with a hyperbolic tangent
activation function, we would adjust the model spec as in the following:

    import org.mitre.mandolin.glp._
    val glp = new GlpModel
    val df = glp.readAsDataFrame(sqlContext, sc, "mnist.10k", 784, 10)
    val data = df.randomSplit(Array(0.8,0.2))
    val trData = data(0)
    val tstData = data(1)
    val mspec = IndexedSeq(LType(InputLType), 
                           LType(TanHLType, 300), 
                           LType(SoftMaxLType))
    val model = glp.estimate(sc, trData, mspec)
    val confusion = glp.evaluate(sc, model, tstData)

The `estimate` method uses stochastic gradient descent to optimize the loss function
as defined by the model specification `mspec`. The default method is **AdaGrad**, which works
well for linear models. Because learning rates can only decrease with AdaGrad, other
adaptive learning rate schemes, such as **RMSProp**, often work better with deep, nonlinear
models having non-convex loss functions.  Alternative parameter updating schemes can be specified
and passed into the `estimate` method:

    val mspec = IndexedSeq(LType(InputLType), 
                           LType(TanHLType, 300), 
                           LType(SoftMaxLType))
    val model = glp.estimate(sc, trData, mspec, RMSPropSpec(0.001))
    val confMat = glp.evaluate(sc, model, tstData)

Note that each `LType` object that describes a layer can provide all the options described
in the [Configuration](configuration.html) section, including dropout, regularization and various loss functions
as different types of output layers.  For example, a deeper model that uses rectified linear
activations in the hidden layer with dropout and L1 regularization could be specified as:

    import org.mitre.mandolin.glp._
    // ...
    val mspec = IndexedSeq(LType(InputLType), 
                           LType(ReluLType, 500, drO=0.5, l1=0.0001), 
                           LType(ReluLType, 700, drO=0.5, l1=0.0001),
                           LType(ReluLType, 500, drO=0.5, l1=0.0001),
                           LType(SoftMaxLType, l1=0.0001))
    val model = glp.estimate(sc, trData, mspec, RMSPropSpec(0.001))
    val confMat = glp.evaluate(sc, model, tstData)


    