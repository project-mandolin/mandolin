import org.mitre.mandolin.ml._
import org.mitre.mandolin.glp._

val glp = new GlpModel // Mandolin class for general GLP model
val glpClassifier = new GLPClassifier(glp) // GLP using spark.ml API

// configure network topology
glpClassifier.setLayerSpec(IndexedSeq(LType(InputLType), LType(SoftMaxLType)))

// read in the data to a spark DataFrame assuming Mandolin input format
// This has a schema with just two columns: (label, features)
// The label is a 0-based double/int denoting the category
// the features column is a Spark feature vector org.apache.spark.mllib.linalg.Vector
val df = glp.readAsDataFrame(sqlContext, sc, "mnist.10k", 784, 10)

// Split the data into a training and test set
val data = df.randomSplit(Array(0.8,0.2))
val tr = data(0)
val tst = data(1)
    
// Fit the model with the training data
val glpClassifierModel = glpClassifier.fit(tr)

// Gather predictions and evaluate the model
// this adds a 'prediction' column to the dataframe
val result = glpClassifierModel.transform(tst)

// Get predictions and labels and evaluate using `spark.ml.evaluation` components
val predictionAndLabels = result.select("prediction","label")
val evaluator = new org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator()
println("F1: " + evaluator.evaluate(predictionAndLabels))
