#!/bin/sh

spark-submit --driver-memory 56G --class org.mitre.mandolin.app.Mandolin ../../dist/mandolin-spark-0.3.5.jar --conf url.model-selection.distributed.conf
