# Installation

Mandolin is written in Scala and grouped into three sub-projects: `mandolin-core`, `mandolin-mx` and `mandolin-spark`.

* `mandolin-core`: This artifact contains Mandolin's core functionality

* `mandolin-mx`: This artifact contains wrappers around MXNet (for deep learning) and XGBoost (for gradient
boosted trees). It relies on pre-built shared objects in the case of MXNet. It depends on `mandolin-core`.

* `mandolin-spark`: This artfact includes functionality for using Apache Spark to speed
up training and/or to provide concurrent model selection using a compute cluster. It depends on
`mandolin-mx`. It also includes some additional functionality that interoperates with `spark.ml`.

All three artifacts can be built from source by downloading [SBT](http://www.scala-sbt.org/download.html).
Builds are platform dependent (due to MXNet and XGBoost relying on native code); currently Linux and Mac
are supported:

Linux
:   @@snip [linux-install.txt](install/linux.txt) 

Mac
:   @@snip [mac-install.txt](install/mac.txt)


This will build three assembly artifacts (i.e. "fat" jar files) :
`mandolin-core-0.3.5.jar`, `mandolin-mx-0.3.5.jar` and `mandolin-spark-0.3.5.jar`.
These are placed in the directory `mandolin/dist`.

If only the the mandolin-core artifact is of interest, just that component can be compiled by executing:

    > sbt "project mandolin-core" assembly

This is especially helpful if MXNet, XGBoost and Spark are not necessary as building those components,
especially Spark, requires a number of libraries to be downloaded.

## Building Mandolin with MXNet and XGBoost

The `mandolin-mx-0.3.5.jar` artifact includes native code pre-compiled for XGBoost depending on the
platform specified in the SBT assembly target (e.g. `linux-assembly` or `osx-assembly`). The native
code for MXNet is ***not*** included, however, as this build is much less compatible across different
linux distributions and assumes various 3rd party libraries are available in the library path.
Further, MXNet is often compiled with GPU support. These complexities prevent providing pre-compiled
versions of MXNet.  Instead, the shared library for MXNet must be compiled separately and designated
at runtime if the intent is to use MXNet. If MXNet is not needed, the `mandolin-mx` and `mandolin-spark`
libraries can still be used with XGBoost or Mandolin's native MLP implementations.  See the
examples for more details.
