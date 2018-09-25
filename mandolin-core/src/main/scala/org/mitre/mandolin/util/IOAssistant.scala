package org.mitre.mandolin.util
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import java.io.DataInputStream

import com.esotericsoftware.kryo.io.{Input => KInput, Output => KOutput}
import com.esotericsoftware.kryo.Kryo

abstract class AbstractPrintWriter(fp: String, append: Boolean) {
  def write(s: String) : Unit
  def write(c: Char) : Unit
  def print(s: String) : Unit
  def print(c: Char) : Unit
  def println : Unit
  def close() : Unit
}

class LocalPrintWriter(fp: String, ap: Boolean) extends AbstractPrintWriter(fp, ap){
    val pr = new java.io.PrintWriter(new java.io.FileOutputStream(new java.io.File(fp), ap))
    def write(s: String) = pr.write(s)
    def write(c: Char)  = pr.write(c)
    def print(s: String)  = pr.print(s)
    def print(c: Char) = pr.print(c)
    def println = pr.println
    def close() = pr.close()
  }
  

abstract class IOAssistant {
  def getPrintWriterFor(filePath: String, append: Boolean): AbstractPrintWriter
  
  def writeSerializedObject(kryo: Kryo, filePath: String, o: Any) : Unit
  def readSerializedObject(kryo: Kryo, filePath: String, c: Class[_]) : Any
  def readSerializedObject(kryo: Kryo, is: DataInputStream, c: Class[_]) : Any
  def readLines(filePath: String): Iterator[String] 
}

class LocalIOAssistant extends IOAssistant {

  def getPrintWriterFor(filePath: String, append: Boolean): AbstractPrintWriter = new LocalPrintWriter(filePath, append)
  
  def readLines(filePath: String): Iterator[String] = {
    scala.io.Source.fromFile(new java.io.File(filePath)).getLines
  }
  

  def writeSerializedObject(kryo: Kryo, filePath: String, o: Any) {    
    val os = new java.io.BufferedOutputStream(new java.io.FileOutputStream(filePath))
    val output = new KOutput(os)
    kryo.writeObject(output, o)
    output.close()
    os.close()    
  }

  def readSerializedObject(kryo: Kryo, filePath: String, c: Class[_]) = {    
    val is = new java.io.BufferedInputStream(new java.io.FileInputStream(filePath))
    val kInput = new KInput(is)
    val m = kryo.readObject(kInput, c)
    kInput.close
    is.close
    m    
  }

  def readSerializedObject(kryo: Kryo, is: DataInputStream, c: Class[_]) = {
    val kInput = new KInput(is)
    val m = kryo.readObject(kInput, c)
    kInput.close
    is.close
    m
  }
}