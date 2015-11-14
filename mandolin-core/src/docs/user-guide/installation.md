{%
  title: Installation
%}

Mandolin is written entirely in Scala. There are two distinct code artifacts, `mandolin-core`
and `mandolin-spark`, the former including the core ML algorithms without any dependency on
Apache Spark while the latter includes Spark-based distributed stochastic optimizers as well
as some additional functionality that interoperates with `spark.ml`.

Both artifacts can be built from source by downloading [SBT](http://www.scala-sbt.org/download.html)
and running:

    > sbt assembly

This will build two assembly artifacts. The artifact 
`mandolin-core/target/scala-2.11/mandolin-core-assembly-0.3_2.11.7.jar` contains Mandolin's machine
learning components without Spark, while 
`mandolin-spark/target/scala-2.11/mandolin-spark-assembly-0.3_2.11.7.jar` contains Mandolin along
with Apache Spark and provides distributed stochastic solvers for the same set of ML algorithms.

The current build targets Scala 2.11. In order to use Mandolin with Apache Spark, the Spark build
must match the major version number of Mandolin.  The current version of Apache Spark is built
against Scala 2.10, however. There are two options: The first is to download Apache Spark and build it against
Scala 2.11 - see docs [here](http://spark.apache.org/docs/latest/building-spark.html#building-for-scala-211).
The second option is to build Mandolin using Scala 2.10.  This can be done by simply executing:

    > sbt 'set scalaVersion := "2.10.5"' assembly

Or, by doing:

    > sbt "+ assembly"

which will build _both_ 2.10.x and 2.11.x versions.

If only the the mandolin-core artifact is of interest, just that component can be compiled by executing:

    > sbt "project mandolin-core" assembly

This will avoid a full build including Spark which can be time-consuming due to the number of 3rd 
party libraries that must be downloaded.