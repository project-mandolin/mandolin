Overview
=========

Mandolin is a tool for training very large-scale prediction models. Scalability
is achieved through the use of distributed computing using [Apache Spark](http://apache.spark.org). 
In its current state, Mandolin is mostly designed to be used as standalone modeling
framework through its use of a flexible configuration system.  
It also contains a developer API, however, for developing new machine learning
algorithms that leverage distributed optimization. 

Mandolin also contains an API extension based on **spark.ml**, the most recent version of
Apache Spark's own [MLLib](http://spark.apache.org/docs/latest/mllib-guide.html) library.
This allows Mandolin models to be used directly within the MLLib ecosystem, leveraging
other datastructures, tools, and MLLib's rich ML Pipeline framework.

The particular ML algorithms that Mandolin provides nearly all fall within a general 
family of learning algorithms known as Multi-Layered Perceptrons, but including a
variety of possible layer configurations, activation functions, output layers with associated
loss functions, and optimizations to handle sparse linear models efficiently.  
