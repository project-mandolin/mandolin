{%
  title: Installation
%}

Mandolin is written entirely in Scala; in order to build to a single .jar file containing
the entire Mandolin distribution, download
[SBT](http://www.scala-sbt.org/download.html) and run the command:

    > sbt assembly

This will build the file `target/scala-2.11/mandolin-assembly-0.2.7-SNAPSHOT_2.11.7.jar`. This .jar
file will contain all the Mandolin code along with all necessary dependencies, i.e. it is self-contained.

The current build targets Scala 2.11. In order to use Mandolin with Apache Spark, the Spark build
must match the major version number of Mandolin.  The current version of Apache Spark is built
against Scala 2.10, however. There are two options: The first is to download Apache Spark and build it against
Scala 2.11 - see docs [here](http://spark.apache.org/docs/latest/building-spark.html#building-for-scala-211).
The second option is to build Mandolin using Scala 2.10.  This can be done by simply executing:

    > sbt 'set scalaVersion := "2.10.5"' assembly

This will build the target jar file in `target/scala-2.11/mandolin-assembly-0.2.7-SNAPSHOT_2.10.5.jar`
Or, by doing:

    > sbt "+ assembly"

which will build _both_ 2.10.x and 2.11.x versions.

<!--

TODO:

  Add build instructions for Spark 1.3 and 1.4

  Decide on default Java target 
  and provide documentation for building to Java 1.7 and 1.8

-->