package org.mitre.mandolin.gm.standalone

import org.mitre.mandolin.gm._
import org.mitre.mandolin.util.{IOAssistant, Alphabet, LocalIOAssistant}
import org.mitre.mandolin.mlp.{MMLPWeights, ANNetwork, CategoricalMMLPPredictor}
import com.twitter.chill.EmptyScalaKryoInstantiator
import org.mitre.mandolin.config.MandolinRegistrator
import com.esotericsoftware.kryo.Kryo

import scalax.collection.mutable.Graph
import scalax.collection.edge.LDiEdge
  
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
  
  
    import scalax.collection.edge.Implicits._
    
  import scalax.collection.edge.LBase._
  
  object MultiFactorLabel extends LEdgeImplicits[MultiFactor]
  import MultiFactorLabel._


  val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)

  def processTrain(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val (fg, alphabets) = FactorGraph.gatherSingleFactorGraph(fgSettings) // just gather a single big Factor Graph for training
    val trainer = new FactorGraphTrainer(fgSettings, List(fg), alphabets)
    val trFg = trainer.trainModels()
    val mWriter = new StandaloneFactorGraphModelWriter
    mWriter.writeModel(io, fgSettings.modelFile.get, trFg.singletonModel.wts, alphabets.sa, alphabets.sla, trainer.singletonNN, 
        trFg.factorModel.fullWts, alphabets.fa, alphabets.fla, trainer.factorNN)
  }

  def processTrainTest(fgSettings: FactorGraphSettings) = {
    val io = new LocalIOAssistant
    val (fg, alphabets) = FactorGraph.gatherSingleFactorGraph(fgSettings)
    val trainer = new FactorGraphTrainer(fgSettings, List(fg), alphabets)
    val trFg = trainer.trainModels()
    val (testFg, _) = FactorGraph.gatherFactorGraphs(fgSettings, Some(alphabets))
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    val runtimeFg = new TrainedFactorGraph(trFg.factorModel, trFg.singletonModel, fgSettings.sgAlpha, infer)
    var wAcc = 0.0
    var cnt = 0
    /*
    val traverser = testFgs.get
    traverser.foreach { (c: Graph[SingletonFactor,LDiEdge]#Component) =>
      val singles = c.nodes.map{_.toOuter}.toVector
      val factors = c.edges.map{_.toOuter} map {case s :~> t + (l: MultiFactor) => l}
      val fg = new FactorGraph(factors.toVector, singles.toVector)          
            
      runtimeFg.mapInfer(fgSettings.subGradEpochs, fg)
      val numSingles = fg.singletons.length
      wAcc += runtimeFg.getAccuracy(fg) * numSingles
      cnt += numSingles
      }
      * 
      */
    logger.info("Test Accuracy: " + (wAcc / cnt))
  }
  
  def decodeNodesEdges(s:Vector[SingletonFactor], f: Vector[MultiFactor], fgSettings: FactorGraphSettings,
      testFgModel: FactorGraphModelSpec, alphabetset: AlphabetSet) = {
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    val outFile = fgSettings.outputFile
    val fg = new FactorGraph(f, s)
      val sn = testFgModel.snet.copy
      val fn = testFgModel.fnet.copy
      
      // need to set up workspace for decoding
      fg.factors foreach {_.setWorkSpace} // this will setup workspace dynamically
      
      val factorModel = new PairFactorModel(fn, sn, testFgModel.fwts, testFgModel.sla.getSize)
      val singleModel = new SingletonFactorModel(new CategoricalMMLPPredictor(sn), testFgModel.swts)
      val runtime = new TrainedFactorGraph(factorModel, singleModel, fgSettings.sgAlpha, infer)
      val numSingles = fg.singletons.length
      logger.info("MAP inference on factor graph with: [ " + numSingles + " singletons and " + fg.factors.length + " pair-wise factors ]")
      runtime.mapInfer(fgSettings.subGradEpochs, fg)
      runtime.renderMapOutput(fg.singletons, outFile.get, alphabetset.sla, true)
      (runtime.getAccuracy(fg), numSingles)
 
  }
  
  def decodeSubGraph(c: Graph[SingletonFactor,LDiEdge]#Component, fgSettings: FactorGraphSettings, 
      testFgModel: FactorGraphModelSpec, alphabetset: AlphabetSet) = {
    val singles = c.nodes.map{_.toOuter}.toVector
    val factors = c.edges.map{_.toOuter} map {case s :~> t + (l: MultiFactor) => l}
    decodeNodesEdges(singles, factors.toVector, fgSettings, testFgModel, alphabetset)                    
  }
  
  def processDecode(fgSettings: FactorGraphSettings) = {
    import scala.collection.parallel._
    val io = new LocalIOAssistant
    val mReader = new StandaloneFactorGraphModelReader
    val testFgModel = mReader.readModel(io, fgSettings.modelFile.get)
    val alphabetset = AlphabetSet(testFgModel.sfa, testFgModel.ffa, testFgModel.sla, testFgModel.fla)
    val (decodeFgs, _) = FactorGraph.gatherFactorGraphs(fgSettings, Some(alphabetset))
    val infer = fgSettings.inferAlgorithm.getOrElse("star")
    
    var wAcc = 0.0
    var cnt = 0
    if (fgSettings.decoderThreads > 1) {
      val batchSize = fgSettings.decoderThreads * 40
      
      val subgraphs = 
        if (fgSettings.factorFile.isDefined) {
      val entireGraph = decodeFgs.get
      val size = entireGraph.nodes.size
      var curSize = size
      while (curSize > 0) {
        var i = 0
        val buf = new collection.mutable.ListBuffer[(Vector[SingletonFactor], Vector[MultiFactor])]
        while (curSize > 0 && i < batchSize) {
          val n = entireGraph.nodes.draw(util.Random)          
          // val nn = entireGraph.nodes.dr
          val c = n.weakComponent
          val singles = c.nodes.map{_.toOuter}.toVector
          val factors = c.edges.map{_.toOuter} map {case s :~> t + (l: MultiFactor) => l}
          buf append ((singles, factors.toVector))
          c.nodes foreach {n => entireGraph -= n; curSize -= 1}
          i += 1
        }
        val parBuf = buf.toList.par      
        parBuf.tasksupport_=(new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(fgSettings.decoderThreads)))
        parBuf foreach {case (n,e) =>
          val (acc, numSingles) = decodeNodesEdges(n, e, fgSettings, testFgModel, alphabetset)
          wAcc += acc * numSingles
          cnt += numSingles
        }
      }
        }
        else {
          val nodes = decodeFgs.get.nodes.toOuter.toVector
          val (acc, numSingles) = decodeNodesEdges(nodes, Vector(), fgSettings, testFgModel, alphabetset)
          wAcc += acc * numSingles
          cnt += numSingles
        }
    } else {
      val entireGraph = decodeFgs.get
      while (entireGraph.size > 0) {
        val n = entireGraph.nodes.draw(util.Random)
        val component = n.weakComponent
        val (acc, numSingles) = decodeSubGraph(component, fgSettings, testFgModel, alphabetset)
        wAcc += acc * numSingles
        cnt += numSingles
        component.nodes foreach {n => entireGraph -= n}
      }
      /*
      val traverser = decodeFgs.get
      logger.info("Traversing graph using single thread ...")
      traverser foreach {(c: Graph[SingletonFactor, LDiEdge]#Component) =>
        logger.info("Processing component ...")
        val (acc, numSingles) = decodeSubGraph(c, fgSettings, testFgModel, alphabetset)
        wAcc += acc * numSingles
        cnt += numSingles
      }
      * 
      */
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