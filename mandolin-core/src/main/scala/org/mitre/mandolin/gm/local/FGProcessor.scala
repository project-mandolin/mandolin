package org.mitre.mandolin.gm.local

import org.mitre.mandolin.gm._
import org.mitre.mandolin.util.{ IOAssistant, Alphabet, LocalIOAssistant }
import org.mitre.mandolin.glp.{GLPWeights, ANNetwork}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar

class LocalRegistrator extends MandolinRegistrator {
  def registerClasses(kryo: Kryo) = register(kryo)
}

class LocalFactorGraphModelWriter extends FactorGraphModelWriter {
  
  val instantiator = new EmptyScalaKryoInstantiator

  val kryo = {
    val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
  }
  
  val registrator = new LocalRegistrator
  registrator.registerClasses(kryo)
  
  def writeModel(io: IOAssistant, filePath: String, 
      sw: GLPWeights, sa: Alphabet, sla: Alphabet, sann: ANNetwork,
      fw: GLPWeights, fa: Alphabet, fla: Alphabet, fann: ANNetwork) : Unit = {
    io.writeSerializedObject(kryo, filePath, FactorGraphModelSpec(sw, fw, sann, fann, sla, fla, sa, fa))
  }
}

class FGProcessor {
  
  def processTrain(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val fg = FactorGraph.gatherFactorGraph(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fg)
    val trFg = trainer.trainModels()    
    val mWriter = new LocalFactorGraphModelWriter
    mWriter.writeModel(io, fgSettings.modelFile.get, trFg.singletonModel.wts, fg.alphabets.sa, fg.alphabets.sla, trainer.singletonNN, 
        trFg.factorModel.wts, fg.alphabets.fa, fg.alphabets.fla, trainer.factorNN)
    
  }
  
  def processTrainTest(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val fg = FactorGraph.gatherFactorGraph(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fg)
    val trFg = trainer.trainModels()    
    val testFg = FactorGraph.gatherFactorGraph(fgSettings, fg.alphabets)
    val runtimeFg = new TrainedFactorGraph(trFg.factorModel, trFg.singletonModel, testFg, fgSettings.sgAlpha)
    runtimeFg.mapInfer(fgSettings.subGradEpochs)
    println("Test Accuracy: " + runtimeFg.getAccuracy)    
  }  
}

object FGProcessor {
  def main(args: Array[String]): Unit = {
    val appSettings = new FactorGraphSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val fgProcessor = new FGProcessor
    mode match {
      case "train"             => fgProcessor.processTrain(appSettings)
      //case "decode"            => localProcessor.processDecode(appSettings)
      case "train-test"        => fgProcessor.processTrainTest(appSettings)
      //case "train-decode"      => localProcessor.processTrainDecode(appSettings)
      //case "train-decode-dirs" => localProcessor.processTrainTestDirectories(appSettings)
    }
    System.exit(0)
  }
}