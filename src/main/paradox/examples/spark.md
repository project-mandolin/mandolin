## Spark

### Web URL

Mandolin's configuration system may also be used to configure Spark jobs
by adding an additional `spark` block to the outermost scope.  This
enables users to set any valid Spark property, just as they would with a
normal Spark job.  Please refer to the Spark documentation for details
on the configuration properties available.

The following config file snippet shows the addition of the `spark`
block.

@@snip [url.distributed.1.conf](snippets/url.distributed.conf)

The application settings block demonstrates how to set typical Spark
application properties such as the application name and Spark master
URL.

The executor settings block requires special attention.  Configuring the
executors correctly is essential to the proper operation of Mandolin.
In this example, we want to run Mandolin on a Spark cluster consisting
of 8 Spark worker nodes each with 16GB of memory, 8 CPU cores, and
running a single executor.  As mentioned in the
@ref[Spark](../spark/spark.md) section of this documentation,
Mandolin employs a trick where the number of cores available to each
executor is forced to 1.  This is realized by setting the
`executor.cores` property to 1. The `cores.max=8` is used to control the
number of executors.  Finally, the amount of RAM is set using the
`executor.memory` property.

Proper configuration of the serialization settings is vital when running
Mandolin on Spark.  The `serializer` and `kryo.registrator` must be set
to the values shown in the snippet.  It is recommended that the Kryo
Serializer buffer size be increased above the default as well.

Distributed Mandolin jobs can be launched using the spark-submit utility as follows:

	   spark-submit --driver-memory 32G --class org.mitre.mandolin.app.Mandolin mandolin-spark-0.3.5.jar --conf url.distributed.conf

Note that the `--driver-memory` option must be specified BEFORE launching the job; it 
cannot be specified in the configuration file.   