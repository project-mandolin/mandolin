#!/bin/sh

spark-submit --driver-memory 40G --class org.mitre.mandolin.glp.spark.Mandolin ../../dist/mandolin-spark-0.3.5.jar --conf url.distributed.conf
