package org.mitre.mandolin.glp
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.optimize.ModelWriter
import com.esotericsoftware.kryo.io.{ Input => KInput, Output => KOutput }
import com.twitter.chill.{ EmptyScalaKryoInstantiator, AllScalaRegistrar }
import org.mitre.mandolin.util.{ Alphabet, IOAssistant }
import org.mitre.mandolin.transform.FeatureExtractor
import org.apache.spark.SparkContext

case class GLPModelSpec(wts: GLPWeights, evaluator: GLPInstanceEvaluator, la: Alphabet, fe: FeatureExtractor[String, GLPFactor]) extends Serializable

/**
 * @author wellner
 */
class GLPModelWriter(sc: Option[SparkContext]) {
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = sc match {
    case Some(s) => 
      val k = new org.apache.spark.serializer.KryoSerializer(s.getConf)      
      k.newKryo()
    case None => 
      val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
  }
  
  val registrator = new org.mitre.mandolin.config.MandolinKryoRegistrator()
  registrator.registerClasses(kryo)

  def writeModel(weights: GLPWeights): Unit = {
    throw new RuntimeException("Intermediate model writing not implemented with GLPWeights")
  }

  def writeModel(io: IOAssistant, filePath: String, w: GLPWeights, la: Alphabet, ev: GLPInstanceEvaluator, fe: FeatureExtractor[String, GLPFactor]): Unit = {
    io.writeSerializedObject(kryo, filePath, GLPModelSpec(w, ev, la, fe))
  }

  /**
   * def writeModel(f: java.io.File, w: GLPWeights, la: Alphabet, ev: GLPInstanceEvaluator, fe: FeatureExtractor[String, GLPFactor]): Unit = {
   * val os = new java.io.BufferedOutputStream(new java.io.FileOutputStream(f))
   * val output = new KOutput(os)
   * kryo.writeObject(output, GLPModelSpec(w, ev, la, fe))
   * output.close()
   * os.close()
   * }
   * def writeModel(uri: String, filePath: String, w: GLPWeights, la: Alphabet, ev: GLPInstanceEvaluator, fe: FeatureExtractor[String, GLPFactor]): Unit = {
   * import org.apache.hadoop.conf.Configuration
   * import org.apache.hadoop.fs.{ FileSystem, Path }
   * val path = new Path(filePath)
   * val conf = new Configuration()
   * conf.set("fs.defaultFS", uri)
   * val fs = FileSystem.get(conf)
   * val os = fs.create(path)
   * val output = new KOutput(os)
   * kryo.writeObject(output, GLPModelSpec(w, ev, la, fe))
   * fs.close()
   * }
   */
}

class GLPModelReader(sc: Option[SparkContext] = None) {
  val instantiator = new EmptyScalaKryoInstantiator
  
  val kryo = sc match {
    case Some(s) => 
      val k = new org.apache.spark.serializer.KryoSerializer(s.getConf)      
      k.newKryo()
    case None => 
      val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
  }

  val registrator = new org.mitre.mandolin.config.MandolinKryoRegistrator()  
  registrator.registerClasses(kryo)

  /**
   * def readModel(f: java.io.File): GLPModelSpec = {
   * val is = new java.io.BufferedInputStream(new java.io.FileInputStream(f))
   * val kInput = new KInput(is)
   * val m = kryo.readObject(kInput, classOf[GLPModelSpec])
   * kInput.close()
   * is.close()
   * m
   * }
   */

  def readModel(f: String, io: IOAssistant): GLPModelSpec = {
    io.readSerializedObject(kryo, f, classOf[GLPModelSpec]).asInstanceOf[GLPModelSpec]
  }
}
