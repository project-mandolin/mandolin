# Installation

Mandolin is written in Scala and grouped into three sub-projects: `mandolin-core`, `mandolin-mx` and `mandolin-spark`.

* `mandolin-core`: This artifact contains Mandolin's core functionality

* `mandolin-mx`: This artifact contains wrappers around MXNet (for deep learning) and XGBoost (for gradient
boosted trees). It relies on pre-built shared objects in the case of MXNet. It depends on `mandolin-core`.

* `mandolin-spark`: This artfact includes functionality leveraging Apache Spark to speed
up training and/or to provide concurrent model selection using a compute cluster. It depends on
`mandolin-mx`. It also includes additional functionality that provides interoperability with `spark.ml`.

All three artifacts can be built from source by downloading [SBT](http://www.scala-sbt.org/download.html).

@@snip [linux-install.txt](install/linux.txt) 

This will build three assembly artifacts (i.e. "fat" jar files) :
`mandolin-core-0.3.5.jar`, `mandolin-mx-0.3.5.jar` and `mandolin-spark-0.3.5.jar`.
These are placed in the directory `mandolin/dist`.

If **only** the the `mandolin-core` artifact is required, it can be compiled by executing:

    > sbt "project mandolin-core" assembly

This is helpful one is training only "native" Mandolin multi-layer perceptron or linear models and
Apache Spark nor XGBoost or MXNet are not needed.  

## Building with MXNet and XGBoost

The `mandolin-mx-0.3.5.jar` artifact provides bindings to both MXNet and XGBoost as well as Mandolin
code that supports model specification via configuration as well as model selection. The native
code for both MXNet and XGBoost is ***not*** included in this artifact, however. Pre-compiled shared
libraries for XGBoost are provided in `mandolin-mx/pre-compiled/xgboost` for both `linux` and `osx`.
A pre-compiled shared library for MXNet on `osx` is available. A pre-built library for MXNet on `linux`
is not provided due to the variety of the builds, e.g. different BLAS implementations, optional GPU support.

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

