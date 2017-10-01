package org.mitre.mandolin.xg

import org.mitre.mandolin.mlp.MMLPFactor
import org.mitre.mandolin.mlp.standalone.Registrator
import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{LocalIOAssistant, IOAssistant,AbstractPrintWriter, Alphabet}
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix,Booster}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.{MandolinRegistrator}
import com.esotericsoftware.kryo.Kryo

/**
 * @author wellner
 */
class XGModelIO {
  
}

case class XGBoostModelSpec(booster: Array[Byte], la: Alphabet, fe: FeatureExtractor[String, MMLPFactor])

abstract class XGBoostModelWriter {
  def writeModel(io: IOAssistant, filePath: String, booster: Booster, la: Alphabet, fe: FeatureExtractor[String, MMLPFactor]) : Unit
}

abstract class XGBoostModelReader {
  def readModel(f: String, io: IOAssistant) : XGBoostModelSpec
}

class StandaloneXGBoostModelWriter extends XGBoostModelWriter {
  
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = {
    val k = instantiator.newKryo()
    k.setClassLoader(Thread.currentThread.getContextClassLoader)
    k
  }

  val registrator = new Registrator
  registrator.registerClasses(kryo)
  
  def writeModel(io: IOAssistant, filePath: String, booster: Booster, la: Alphabet, fe: FeatureExtractor[String, MMLPFactor]) : Unit = {
    val boosterBytes = booster.toByteArray
    io.writeSerializedObject(kryo, filePath, XGBoostModelSpec(boosterBytes, la, fe))
  }
  
}

class StandaloneXGBoostModelReader extends XGBoostModelReader {
  val instantiator = new EmptyScalaKryoInstantiator

  val registrator = new Registrator

  val kryo = {
    val k = instantiator.newKryo()
    k.setClassLoader(Thread.currentThread.getContextClassLoader)
    k
  }

  registrator.registerClasses(kryo)

  def readModel(f: String, io: IOAssistant) : XGBoostModelSpec = {
    io.readSerializedObject(kryo, f, classOf[XGBoostModelSpec]).asInstanceOf[XGBoostModelSpec]
  }
}