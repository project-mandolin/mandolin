# Installation

Mandolin is written in Scala and grouped into three sub-projects: `mandolin-core`, `mandolin-mx` and `mandolin-spark`.

* `mandolin-core`: This artifact contains Mandolin's core functionality

* `mandolin-mx`: This artifact contains wrappers around MXNet (for deep learning) and XGBoost (for gradient
boosted trees). Both of these libraries use native code. Currently, Mandolin builds artifacts without any
bundled native code and assumes the user will provide the appropriate shared libraries at runtime.
This artifact depends on `mandolin-core`. 

* `mandolin-spark`: This artfact includes functionality leveraging Apache Spark to speed
up training and/or to provide concurrent model selection using a compute cluster. It depends on
`mandolin-mx`. It also includes additional functionality that provides interoperability with `spark.ml`.

All three artifacts can be built from source by downloading [SBT](http://www.scala-sbt.org/download.html).

    > sbt assembly

This will build three assembly artifacts (i.e. "fat" jar files) :
`mandolin-core-0.3.6.jar`, `mandolin-mx-0.3.6.jar` and `mandolin-spark-0.3.6.jar`.
These are placed in the directory `mandolin/dist`.

If **only** the the `mandolin-core` artifact is required, it can be compiled by executing:

    > sbt "project mandolin-core" assembly

This is helpful if one is using only Mandolin's built-in  multi-layer perceptron or linear models and
Apache Spark nor XGBoost or MXNet are not needed.  

## Building with MXNet and XGBoost

The `mandolin-mx-0.3.6.jar` artifact provides bindings to both MXNet and XGBoost as well as Mandolin
code that supports model specification via configuration as well as model selection. The native
code for both MXNet and XGBoost is ***not*** included in this artifact, however. Pre-compiled shared
libraries for XGBoost are provided in `mandolin-mx/pre-compiled/xgboost` for both `linux` and `osx`.
A pre-compiled shared library for MXNet on `osx` is available. A pre-built library for MXNet on `linux`
is not provided due to the variety of the builds, e.g. different BLAS implementations, optional GPU support.

### Rebuilding MXNet and/or XGBoost from scratcth

To use Mandolin with new versions of XGBoost or MXNet, follow the steps below:

  1) Build XGBoost and/or MXNet shared libraries *and* the Scala bindings.  For MXNet, only build the *core*
     bindings that do not include any native code embedded in the resulting `.jar` file.
  2) Remove any existing .jar files in the directory `mandolin/mandolin-mx/lib` and place the newly built .jar files in their place.
  3) Build Mandolin using the steps outlined above (i.e. `sbt assembly` from the top-level directory).
     

## Mandolin with Spark

Currently the Spark build `mandolin-spark` depends on the `mandolin-mx` artifact. It provides fast
distributed training of individual models using a Spark cluster; it also enables fast distributed
model selection.  As with the `mandolin-mx` artifact, the native code for MXNet and/or XGBoost
must be specified at runtime.

## Using MXNet and XGBoost

Runtime libraries must be included in the library path available ot the JVM. This can be done via
the `LD_LIBRARY_PATH` environment variable or using the option `Djava.library.path=<path>` provided
on the command line when invoking the JVM.  When using Spark via the `spark-submit` script, various
options are available to provide shared libraries at runtime; see the Apache Spark documentation
as well as the section documenting Mandolin examples using Spark.

