import org.mitre.mandolin.ml._
import org.mitre.mandolin.mlp._

val mlp = new MMLPModel // Mandolin class for general MLP model
val mlpClassifier = new MMLPClassifier(mlp) // MMLP using spark.ml API

// configure network topology
mlpClassifier.setLayerSpec(IndexedSeq(LType(InputLType), LType(SoftMaxLType)))

// read in the data to a spark DataFrame assuming Mandolin input format
// This has a schema with just two columns: (label, features)
// The label is a 0-based double/int denoting the category
// the features column is a Spark feature vector org.apache.spark.mllib.linalg.Vector
val df = mlp.readAsDataFrame(sqlContext, sc, "mnist.10k", 784, 10)

// Split the data into a training and test set
val data = df.randomSplit(Array(0.8,0.2))
val tr = data(0)
val tst = data(1)
    
// Fit the model with the training data
val mlpClassifierModel = mlpClassifier.fit(tr)

// Gather predictions and evaluate the model
// this adds a 'prediction' column to the dataframe
val result = mlpClassifierModel.transform(tst)

// Get predictions and labels and evaluate using `spark.ml.evaluation` components
val predictionAndLabels = result.select("prediction","label")
val evaluator = new org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator()
println("F1: " + evaluator.evaluate(predictionAndLabels))
