package org.mitre.mandolin.gm

import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{ StdAlphabet, IdentityAlphabet, Alphabet, DenseTensor1 => DenseVec }
import org.mitre.mandolin.glp.{ StdGLPFactor, ANNetwork, LType, InputLType, SoftMaxLType, GLPFactor, GLPWeights, GLPLossGradient, 
  GLPPredictor, GLPAdaGradUpdater, GLPInstanceEvaluator }
import org.mitre.mandolin.glp.local.LocalGLPOptimizer
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.predict.local.LocalTrainer

case class AlphabetSet(sa: Alphabet, fa: Alphabet, sla: Alphabet, fla: Alphabet)

class FactorGraphTrainer {
  
  def getOptimizer(fgSettings: FactorGraphSettings, network: ANNetwork) = {
    LocalGLPOptimizer.getLocalOptimizer(fgSettings, network)
  }
  
  def getOptimizer(network: ANNetwork) = {
    val weights = network.generateRandomWeights
    val sumSquared = network.generateZeroedLayout
    sumSquared set 0.1f // set to the initial learning rate
    val updater = new GLPAdaGradUpdater(sumSquared, 0.1f, None, l1Array = None, l2Array = None)
    val evaluator = new GLPInstanceEvaluator[GLPAdaGradUpdater](network)
    new LocalOnlineOptimizer[GLPFactor, GLPWeights, GLPLossGradient, GLPAdaGradUpdater](weights, evaluator, updater,10,1, 1,None)    
  }
  
  def getComponents(fgSettings: FactorGraphSettings, as: AlphabetSet) = {
    val fgProcessor = new FactorGraphProcessor
    val singletonSp = fgProcessor.getGLPSpec(fgSettings.netspec, as.sa.getSize, as.sla.getSize)
    val factorSp   = fgProcessor.getGLPSpec(fgSettings.netspec, as.fa.getSize, as.fla.getSize)
    //val singletonAnn = ANNetwork(IndexedSeq(LType(InputLType, as.sa.getSize), LType(SoftMaxLType,as.sla.getSize)))
    //val factorAnn    = ANNetwork(IndexedSeq(LType(InputLType, as.fa.getSize), LType(SoftMaxLType,as.fla.getSize)))
    (ANNetwork(singletonSp), ANNetwork(factorSp))    
  }
  
  def trainModels(fgSettings: FactorGraphSettings, factorGraph: FactorGraph) = {
    val (singletonNN, factorNN) = getComponents(fgSettings, factorGraph.alphabets)
    val sOpt = getOptimizer(fgSettings, singletonNN)
    val fOpt = getOptimizer(fgSettings, factorNN)
    val ie = new IdentityExtractor(new IdentityAlphabet(1))
    val sTrain = new LocalTrainer(ie, sOpt)
    val fTrain = new LocalTrainer(ie, fOpt)
    val (sWeights,_) = sTrain.trainWeights(factorGraph.singletons)
    val (fWeights,_) = fTrain.trainWeights(factorGraph.factors)
    val sm = new FactorModel(new GLPPredictor(singletonNN, true), sWeights)
    val fm = new FactorModel(new GLPPredictor(factorNN, true), fWeights)
    new TrainedFactorGraph(fm, sm, factorGraph)
  }
  
}

class FactorGraph(val factors: Vector[MultiFactor], val singletons: Vector[SingletonFactor], val alphabets: AlphabetSet)

class TrainedFactorGraph(val factorModel: FactorModel, val singletonModel: FactorModel, _fs: Vector[MultiFactor], _ss: Vector[SingletonFactor], _a: AlphabetSet) 
extends FactorGraph(_fs, _ss, _a) {
  def this(fm: FactorModel, sm: FactorModel, fg: FactorGraph) = this(fm, sm, fg.factors, fg.singletons, fg.alphabets)
  
  val subGradInference = new SubgradientInference(factorModel, singletonModel)
  
  def mapInfer(n: Int) = {
    subGradInference.mapInfer(factors, singletons, n)
  }
  
  def getMap() = {
    singletons map {s => s.getMode(singletonModel, true)}
  }
}

object FactorGraph {
  
  def getLineInfo(rest: List[String], alphabet: Alphabet, buildVecs: Boolean) = {
      val fvbuf = new collection.mutable.ArrayBuffer[Feature]
      rest foreach { f =>
          f.split(':').toList match {
            case a :: b :: Nil =>
                if (buildVecs) {
                  val bd = b.toDouble
                  val id = alphabet.ofString(a, bd)
                  if (id >= 0) fvbuf append (new NonUnitFeature(alphabet.ofString(a, bd), bd))
                } else alphabet.ofString(a, b.toDouble)
            case a :: Nil =>
                if (buildVecs) {
                  val id = alphabet.ofString(a)
                  if (id >= 0) fvbuf append (new Feature(id))
                } else alphabet.ofString(a)
            case a => throw new RuntimeException("Unparsable feature: " + a)
          }
        }
      fvbuf.toArray    
    } 
  
  
  def getFeatureAlphabetSingleton(fstr: String) : Alphabet = {
    val ifile = new java.io.File(fstr)
    val fa = new StdAlphabet()
    val lines = scala.io.Source.fromFile(ifile).getLines
    lines foreach {l =>
      val p = l.split(' ').toList
      val lab = p.head
      val rest = p.tail   
      FactorGraph.getLineInfo(rest.tail, fa,false)      
    }
    fa
  }
  
  def getFeatureAlphabetFactor(fstr: String) : Alphabet = {
    val ifile = new java.io.File(fstr)
    val fa = new StdAlphabet()
    val lines = scala.io.Source.fromFile(ifile).getLines
    lines foreach {l =>
      val p = l.split(' ').toList
      val lab = p.head
      val rest = p.tail   
      FactorGraph.getLineInfo(rest, fa,false)      
    }
    fa
  }

  def gatherFactorGraph(singletonStr: String, factorStr: String, cardinality: Int) : FactorGraph = {
    val sa = getFeatureAlphabetSingleton(singletonStr)
    val fa = getFeatureAlphabetFactor(factorStr)
    val sla = new IdentityAlphabet(cardinality, false, true)
    val fla = new IdentityAlphabet(cardinality*cardinality, false, true)
    val singletonExtractor = new GMSingletonExtractor(sa, sla)
    val singletons = scala.io.Source.fromFile(new java.io.File(singletonStr)).getLines.toList map singletonExtractor.extractFeatures
    val factorExtractor = new GMFactorExtractor(fa, fla, cardinality,singletonExtractor.variableToSingletonFactors)
    val factors = scala.io.Source.fromFile(new java.io.File(factorStr)).getLines.toList map factorExtractor.extractFeatures
    new FactorGraph(factors.toVector, singletons.toVector, AlphabetSet(sa, fa, sla, fla))
  }  
}

class IdentityExtractor(alphabet: Alphabet) extends FeatureExtractor[GMFactor, GLPFactor] {
  def getAlphabet = alphabet
  def getNumberOfFeatures = alphabet.getSize
  def extractFeatures(fm: GMFactor) : GLPFactor = fm.getInput
}

class GMFactorExtractor(factorAlphabet: Alphabet, factorLa: Alphabet, varOrder: Int, varToSingles: Map[String, SingletonFactor]) 
extends FeatureExtractor[String, MultiFactor] {
  
  val sep = ' '
  val buildVecs = true
  
  def getAlphabet = factorAlphabet
  
  def getNumberOfFeatures = factorAlphabet.getSize

  def extractFeatures(s: String) : MultiFactor = {
    
    val p = s.split(sep).toList
    val lab = p.head
    val rest = p.tail
            
    val lb = lab.split('-').toList
    lb match {
      case v1 :: v2 :: Nil => 
        val fv = FactorGraph.getLineInfo(rest, factorAlphabet,true)        
        val dVec : DenseVec = DenseVec.zeros(factorAlphabet.getSize)    
        fv foreach { f =>
          if (f.fid >= 0) {
            val fv = factorAlphabet.getValue(f.fid, f.value).toFloat
            dVec.update(f.fid, fv)
          }
        }
        val s1 = varToSingles(v1)
        val s2 = varToSingles(v2)        
        val sgs = Array(s1, s2)
        val mf = new MultiFactor(varOrder, sgs, dVec, (v1+v2))
        s1.addParent(mf, 0)
        s2.addParent(mf, 1)
        mf
      case a => throw new RuntimeException("Invalid input: " + a)
    }
  }
}

class GMSingletonExtractor(singletonAlphabet: Alphabet, singletonLa: Alphabet) extends FeatureExtractor[String, SingletonFactor] {
  
  val sep = ' '
  val buildVecs = true
  

  var variableToSingletonFactors : Map[String, SingletonFactor] = Map[String, SingletonFactor]()
  
  def getAlphabet = singletonAlphabet
  
  def getNumberOfFeatures = singletonAlphabet.getSize
  
  
  def extractFeatures(s: String) : SingletonFactor = {
    val p = s.split(sep).toList
    val lab = p.head
    val rest = p.tail
        
    val lb = lab.split('-').toList
    lb match {
      case v1 :: Nil =>
        val label = rest.head
        val fstrs = rest.tail
        val fv = FactorGraph.getLineInfo(fstrs, singletonAlphabet,true)
        val dVec : DenseVec = DenseVec.zeros(singletonAlphabet.getSize)    
        fv foreach { f =>
          if (f.fid >= 0) {
            val fv = singletonAlphabet.getValue(f.fid, f.value).toFloat
            dVec.update(f.fid, fv)
          }
        }
        val l_ind = singletonLa.ofString(label)

        val lv = DenseVec.zeros(singletonLa.getSize)
        lv.update(l_ind,1.0f) // one-hot encoding
        val sgf = new StdGLPFactor(-1, dVec, lv, None)
        val factor = new SingletonFactor(sgf, l_ind)
        variableToSingletonFactors += (v1 -> factor)
        factor
      
      case _ => throw new RuntimeException("Invalid input: " + s)
    }
  }
  
}