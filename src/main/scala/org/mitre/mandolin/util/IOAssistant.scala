package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.apache.spark.SparkContext
import org.apache.hadoop.fs.{ FileSystem, Path, FSDataOutputStream }
import org.apache.hadoop.conf.Configuration
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.esotericsoftware.kryo.Kryo

class IOAssistant(sc: Option[SparkContext]) {
  def this() = this(None)

  def getSparkContext : Option[SparkContext] = sc

  class PrintWriter(filePath: String, append: Boolean) {
    var hdfsOS: FSDataOutputStream = null
    var fs: FileSystem = null
    var pr: java.io.PrintWriter = null

    if (sc.isDefined) {
      if (filePath.startsWith("hdfs://") || (sc.get.master equals "yarn-cluster")) {
        val path = new Path(filePath)
        val conf = new Configuration
        fs = FileSystem.get(new java.net.URI(filePath), conf)
        hdfsOS = if (append) fs.append(path) else fs.create(path)
      } else {
        pr = new java.io.PrintWriter(new java.io.FileOutputStream(new java.io.File(filePath), append))
      }
    } else {
      pr = new java.io.PrintWriter(new java.io.FileOutputStream(new java.io.File(filePath), append))
    }

    def write(s: String) {
      if (hdfsOS != null) hdfsOS.writeChars(s)
      else if (pr != null) pr.write(s)
    }

    def write(c: Char) {
      if (hdfsOS != null) hdfsOS.writeChar(c)
      else if (pr != null) pr.write(c)
    }

    def print(s: String) {
      if (hdfsOS != null) hdfsOS.writeChars(s)
      else if (pr != null) pr.print(s)
    }

    def print(c: Char) {
      if (hdfsOS != null) hdfsOS.writeChar(c)
      else if (pr != null) pr.print(c)
    }

    def println {
      if (hdfsOS != null) hdfsOS.writeChar('\n')
      else if (pr != null) pr.println
    }

    def close() {
      if (hdfsOS != null) hdfsOS.close
      if (fs != null) fs.close
      if (pr != null) pr.close
    }

  }

  def readLines(filePath: String): Iterator[String] = {
    if ((sc.isDefined) && (filePath.startsWith("hdfs://") || (sc.get.master equals "yarn-cluster"))) {
        val path = new Path(filePath)
        val conf = new Configuration
        //val fs = FileSystem.get(sc.get.hadoopConfiguration)
        val fs = FileSystem.get(new java.net.URI(filePath), conf)
        val is = fs.open(path)
        val lines = Stream.continually(is.readLine()).takeWhile(_ != null).toIterator
        //is.close
        //fs.close
        lines
      }    
    else scala.io.Source.fromFile(new java.io.File(filePath)).getLines
  }

  def getPrintWriterFor(filePath: String, append: Boolean): PrintWriter = new PrintWriter(filePath, append)

  def writeSerializedObject(kryo: Kryo, filePath: String, o: Any) {    
    if ((sc.isDefined) && (filePath.startsWith("hdfs://") || (sc.get.master equals "yarn-cluster"))) {
        val path = new Path(filePath)
        val fs = FileSystem.get(sc.get.hadoopConfiguration)
        val os = fs.create(path)
        val output = new KOutput(os)
        kryo.writeObject(output, o)
        output.close
        os.close
      
    } else {
    val os = new java.io.BufferedOutputStream(new java.io.FileOutputStream(filePath))
    val output = new KOutput(os)
    kryo.writeObject(output, o)
    output.close()
    os.close()
    }
  }

  def readSerializedObject(kryo: Kryo, filePath: String, c: Class[_]) = {
    if ((sc.isDefined) && ((filePath.startsWith("hdfs://") || (sc.get.master equals "yarn-cluster")))) {
        val path = new Path(filePath)
        val conf = new Configuration
        val fs = FileSystem.get(new java.net.URI(filePath), conf)
        val is = fs.open(path)
        val kInput = new KInput(is)
        val m = kryo.readObject(kInput, c)
        kInput.close
        is.close
        m  
    } else {
    val is = new java.io.BufferedInputStream(new java.io.FileInputStream(filePath))
    val kInput = new KInput(is)
    val m = kryo.readObject(kInput, c)
    kInput.close
    is.close
    m
    }
  }
}