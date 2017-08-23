# Spark

@@@ index

 * [Examples](examples/examples.md)

@@@

Mandolin can utilize Apache Spark for large-scale learning tasks.  The Spark-based APIs
are present in the mandolin-spark artifact.  Largely all of the Mandolin and Spark
integration centers around Mandolin's distributed stochastic gradient descent implementation.
This means that only minor changes to the configuration file and runtime invocation are required,
while the model specification and input formats do not change from Mandolin's standalone mode.




## MLLib Integration

