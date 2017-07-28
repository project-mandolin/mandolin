package org.mitre.mandolin.gm.local

import org.mitre.mandolin.gm._
import org.mitre.mandolin.util.{ IOAssistant, Alphabet, LocalIOAssistant }
import org.mitre.mandolin.glp.{GLPWeights, ANNetwork, CategoricalGLPPredictor}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar

class LocalRegistrator extends MandolinRegistrator {
  def registerClasses(kryo: Kryo) = register(kryo)
}

trait KryoSetup {
  
  def setupKryo() = {
    val instantiator = new EmptyScalaKryoInstantiator
    val kryo = {
    val k = instantiator.newKryo()
      k.setClassLoader(Thread.currentThread.getContextClassLoader)
      k
    }
  
    val registrator = new LocalRegistrator
    registrator.registerClasses(kryo)
    kryo
  }
} 

class LocalFactorGraphModelWriter extends FactorGraphModelWriter with KryoSetup {
  
  val kryo = setupKryo()
  
  def writeModel(io: IOAssistant, filePath: String, 
      sw: GLPWeights, sa: Alphabet, sla: Alphabet, sann: ANNetwork,
      fw: MultiFactorWeights, fa: Alphabet, fla: Alphabet, fann: ANNetwork) : Unit = {
    io.writeSerializedObject(kryo, filePath, FactorGraphModelSpec(sw, fw, sann, fann, sla, fla, sa, fa))
  }
}

class LocalFactorGraphModelReader extends FactorGraphModelReader with KryoSetup {

  val kryo = setupKryo()
  def readModel(io: IOAssistant, filePath: String) : FactorGraphModelSpec = {
    io.readSerializedObject(kryo, filePath, classOf[FactorGraphModelSpec]).asInstanceOf[FactorGraphModelSpec]
  }
}


class FGProcessor {
  
  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  def processTrain(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val fg = FactorGraph.gatherFactorGraph(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fg)
    val trFg = trainer.trainModels()    
    val mWriter = new LocalFactorGraphModelWriter
    mWriter.writeModel(io, fgSettings.modelFile.get, trFg.singletonModel.wts, fg.alphabets.sa, fg.alphabets.sla, trainer.singletonNN, 
        trFg.factorModel.fullWts, fg.alphabets.fa, fg.alphabets.fla, trainer.factorNN)
    
  }
  
  def processTrainTest(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val fg = FactorGraph.gatherFactorGraph(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fg)
    val trFg = trainer.trainModels()    
    val testFg = FactorGraph.gatherFactorGraph(fgSettings, fg.alphabets)
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    val runtimeFg = new TrainedFactorGraph(trFg.factorModel, trFg.singletonModel, testFg, fgSettings.sgAlpha, infer)
    runtimeFg.mapInfer(fgSettings.subGradEpochs)
    logger.info("Test Accuracy: " + runtimeFg.getAccuracy)    
  }  
  
  def processDecode(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val mReader = new LocalFactorGraphModelReader
    val testFgModel = mReader.readModel(io, fgSettings.modelFile.get)
    val factorModel = new PairFactorModel(testFgModel.fnet, testFgModel.snet, testFgModel.fwts)
    val singleModel = new SingletonFactorModel(new CategoricalGLPPredictor(testFgModel.snet), testFgModel.swts)
    val alphabetset = AlphabetSet(testFgModel.sfa, testFgModel.ffa, testFgModel.sla, testFgModel.fla)
    val decodeFg = FactorGraph.gatherFactorGraph(fgSettings, alphabetset)
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    val runtime = new TrainedFactorGraph(factorModel, singleModel, decodeFg, fgSettings.sgAlpha, infer)
    runtime.mapInfer(fgSettings.subGradEpochs)
    val outFile = fgSettings.outputFile
    runtime.renderMapOutput(outFile.get, testFgModel.sla)
    logger.info("Test Accuracy: " + runtime.getAccuracy)
  }
}

object FGProcessor extends org.mitre.mandolin.config.LogInit {
  def main(args: Array[String]): Unit = {
    val appSettings = new FactorGraphSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val fgProcessor = new FGProcessor
    mode match {
      case "train"             => fgProcessor.processTrain(appSettings)
      case "decode"            => fgProcessor.processDecode(appSettings)
      case "train-test"        => fgProcessor.processTrainTest(appSettings)
      //case "train-decode"      => localProcessor.processTrainDecode(appSettings)
      //case "train-decode-dirs" => localProcessor.processTrainTestDirectories(appSettings)
    }
    System.exit(0)
  }
}