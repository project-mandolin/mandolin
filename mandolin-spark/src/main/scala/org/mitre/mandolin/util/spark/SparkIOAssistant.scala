package org.mitre.mandolin.util.spark

import java.io.DataInputStream

import org.apache.spark.SparkContext
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.conf.Configuration
import com.esotericsoftware.kryo.io.{Input => KInput, Output => KOutput}
import com.esotericsoftware.kryo.Kryo
import org.mitre.mandolin.util.{AbstractPrintWriter, IOAssistant, LocalPrintWriter}
/**
 * @author wellner
 */

object PrintWriter {
  def apply(fp: String, ap: Boolean) = new LocalPrintWriter(fp, ap)
  def apply(fp: String, ap: Boolean, sc: SparkContext) = {
    if (fp.startsWith("hdfs://") || (sc.master equals "yarn-cluster"))
      new PrintWriter(fp, ap, sc)
    else new LocalPrintWriter(fp, ap)
  }
}

class PrintWriter(filePath: String, append: Boolean, sc: SparkContext) extends AbstractPrintWriter(filePath, append) {    
    val path = new Path(filePath)
    val conf = new Configuration
    val fs = FileSystem.get(new java.net.URI(filePath), conf)
    val hdfsOS = if (append) fs.append(path) else fs.create(path)

    def write(s: String) = hdfsOS.writeChars(s)

    def write(c: Char) = hdfsOS.writeChar(c)

    def print(s: String) = hdfsOS.writeChars(s)

    def print(c: Char) = hdfsOS.writeChar(c)

    def println = hdfsOS.writeChar('\n')

    def close() = hdfsOS.close

  }

class SparkIOAssistant(sc: SparkContext) extends IOAssistant {

  def getPrintWriterFor(filePath: String, append: Boolean): AbstractPrintWriter = PrintWriter(filePath, append, sc)

  def readLines(filePath: String): Iterator[String] = {
    if ((filePath.startsWith("hdfs://") || (sc.master equals "yarn-cluster"))) {
      val path = new Path(filePath)
      val conf = new Configuration
      val fs = FileSystem.get(new java.net.URI(filePath), conf)
      val is = fs.open(path)
      val lines = Stream.continually(is.readLine()).takeWhile(_ != null).toIterator
      lines
    } else scala.io.Source.fromFile(new java.io.File(filePath)).getLines
  }

  def writeSerializedObject(kryo: Kryo, filePath: String, o: Any) {
    if ((filePath.startsWith("hdfs://") || (sc.master equals "yarn-cluster"))) {
      val path = new Path(filePath)
      val fs = FileSystem.get(sc.hadoopConfiguration)
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
    if (((filePath.startsWith("hdfs://") || (sc.master equals "yarn-cluster")))) {
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

  def readSerializedObject(kryo: Kryo, is: DataInputStream, c: Class[_]): Any = throw new RuntimeException("not implemented")
}