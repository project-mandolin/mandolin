package org.mitre.mandolin.gm.standalone

import org.mitre.mandolin.gm._
import org.mitre.mandolin.util.{IOAssistant, Alphabet, LocalIOAssistant}
import org.mitre.mandolin.mlp.{MMLPWeights, ANNetwork, CategoricalMMLPPredictor}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo

class StandaloneRegistrator extends MandolinRegistrator {
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

    val registrator = new StandaloneRegistrator
    registrator.registerClasses(kryo)
    kryo
  }
}

class StandaloneFactorGraphModelWriter extends FactorGraphModelWriter with KryoSetup {

  val kryo = setupKryo()
  def writeModel(io: IOAssistant, filePath: String,
      sw: MMLPWeights, sa: Alphabet, sla: Alphabet, sann: ANNetwork,
      fw: MultiFactorWeights, fa: Alphabet, fla: Alphabet, fann: ANNetwork) : Unit = {
    io.writeSerializedObject(kryo, filePath, FactorGraphModelSpec(sw, fw, sann, fann, sla, fla, sa, fa))
  }
}

class StandaloneFactorGraphModelReader extends FactorGraphModelReader with KryoSetup {

  val kryo = setupKryo()

  def readModel(io: IOAssistant, filePath: String): FactorGraphModelSpec = {
    io.readSerializedObject(kryo, filePath, classOf[FactorGraphModelSpec]).asInstanceOf[FactorGraphModelSpec]
  }
}


class FGProcessor {

  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def processTrain(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val (fgs, alphabets) = FactorGraph.gatherFactorGraphs(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fgs, alphabets)
    val trFg = trainer.trainModels()
    val mWriter = new StandaloneFactorGraphModelWriter
    mWriter.writeModel(io, fgSettings.modelFile.get, trFg.singletonModel.wts, alphabets.sa, alphabets.sla, trainer.singletonNN, 
        trFg.factorModel.fullWts, alphabets.fa, alphabets.fla, trainer.factorNN)
  }

  def processTrainTest(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val (fgs, alphabets) = FactorGraph.gatherFactorGraphs(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, fgs, alphabets)
    val trFg = trainer.trainModels()
    val (testFgs, _) = FactorGraph.gatherFactorGraphs(fgSettings, alphabets)
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    val runtimeFg = new TrainedFactorGraph(trFg.factorModel, trFg.singletonModel, fgSettings.sgAlpha, infer)
    var wAcc = 0.0
    var cnt = 0
    testFgs foreach { fg => 
      runtimeFg.mapInfer(fgSettings.subGradEpochs, fg)
      val numSingles = fg.singletons.length
      wAcc += runtimeFg.getAccuracy(fg) * numSingles
      cnt += numSingles
      }
    logger.info("Test Accuracy: " + (wAcc / cnt))
  }
  
  def processDecode(fgSettings: FactorGraphSettings) = {
    import scala.collection.parallel._
    val io = new LocalIOAssistant
    val mReader = new StandaloneFactorGraphModelReader
    val testFgModel = mReader.readModel(io, fgSettings.modelFile.get)
    val alphabetset = AlphabetSet(testFgModel.sfa, testFgModel.ffa, testFgModel.sla, testFgModel.fla)
    val (decodeFgs, _) = FactorGraph.gatherFactorGraphs(fgSettings, alphabetset)
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    
    var wAcc = 0.0
    var cnt = 0
    val outFile = fgSettings.outputFile
    val parFgs = decodeFgs.par
    logger.info("Decoding " + decodeFgs.length + " graphs using " + fgSettings.decoderThreads + " compute threads")
    parFgs.tasksupport_=(new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(fgSettings.decoderThreads)))
    parFgs foreach { fg =>
      val sn = testFgModel.snet.copy
      val fn = testFgModel.fnet.copy
      val factorModel = new PairFactorModel(fn, sn, testFgModel.fwts, testFgModel.sla.getSize)
      val singleModel = new SingletonFactorModel(new CategoricalMMLPPredictor(sn), testFgModel.swts)
      val runtime = new TrainedFactorGraph(factorModel, singleModel, fgSettings.sgAlpha, infer)
      val numSingles = fg.singletons.length
      logger.info("MAP inference on factor graph with: [ " + numSingles + " singletons and " + fg.factors.length + " pair-wise factors ]")
      runtime.mapInfer(fgSettings.subGradEpochs, fg)
      runtime.renderMapOutput(fg.singletons, outFile.get, alphabetset.sla, true)
      
      wAcc += runtime.getAccuracy(fg) * numSingles
      cnt += numSingles
      }
    logger.info("Test Accuracy: " + (wAcc / cnt))    
  }
}

object FGProcessor extends org.mitre.mandolin.config.LogInit {
  def main(args: Array[String]): Unit = {
    val appSettings = new FactorGraphSettings(args)
    val mode = appSettings.appMode
    val numPartitions = appSettings.numPartitions
    val fgProcessor = new FGProcessor
    mode match {
      case "train" => fgProcessor.processTrain(appSettings)
      case "decode" => fgProcessor.processDecode(appSettings)
      case "train-test" => fgProcessor.processTrainTest(appSettings)
    }
    System.exit(0)
  }
}