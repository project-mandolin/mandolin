package org.mitre.mandolin.gm

import org.mitre.mandolin.transform.FeatureExtractor
import org.mitre.mandolin.util.{ StdAlphabet, IdentityAlphabet, Alphabet, DenseTensor1 => DenseVec }
import org.mitre.mandolin.glp.{ StdGLPFactor, ANNetwork, ANNBuilder, LType, InputLType, SoftMaxLType, GLPFactor, GLPWeights, GLPLossGradient, 
  CategoricalGLPPredictor, GLPAdaGradUpdater, GLPInstanceEvaluator }
import org.mitre.mandolin.glp.local.LocalGLPOptimizer
import org.mitre.mandolin.optimize.local.LocalOnlineOptimizer
import org.mitre.mandolin.predict.local.LocalTrainer
import org.slf4j.LoggerFactory

case class AlphabetSet(sa: Alphabet, fa: Alphabet, sla: Alphabet, fla: Alphabet)

class FactorGraphTrainer(fgSettings: FactorGraphSettings, factorGraph: FactorGraph) {
  
  val logger = LoggerFactory.getLogger(this.getClass)
  
  val (singletonNN, factorNN) = getComponents(fgSettings, factorGraph.alphabets)
  val sOpt = LocalGLPOptimizer.getLocalOptimizer(fgSettings, singletonNN)
  val fOpt = LocalGLPOptimizer.getLocalOptimizer(fgSettings, factorNN)
  val ie = new IdentityExtractor(new IdentityAlphabet(1))
  val sTrainer = new LocalTrainer(ie, sOpt)
  val fTrainer = new LocalTrainer(ie, fOpt)
  
  def getComponents(fgSettings: FactorGraphSettings, as: AlphabetSet) = {
    val fgProcessor = new FactorGraphProcessor
    val singletonSp = ANNBuilder.getGLPSpec(fgSettings.netspec, as.sa.getSize, as.sla.getSize)
    val factorSp   = ANNBuilder.getGLPSpec(fgSettings.netspec, as.fa.getSize, as.fla.getSize)
    //val singletonAnn = ANNetwork(IndexedSeq(LType(InputLType, as.sa.getSize), LType(SoftMaxLType,as.sla.getSize)))
    //val factorAnn    = ANNetwork(IndexedSeq(LType(InputLType, as.fa.getSize), LType(SoftMaxLType,as.fla.getSize)))
    (ANNetwork(singletonSp), ANNetwork(factorSp))    
  }
  
  def trainModels() = {
    val t = System.nanoTime
    logger.info("Estimating singleton parameters ...")
    val (sWeights,_) = sTrainer.trainWeights(factorGraph.singletons)
    logger.info("Estimation finished in " + ((System.nanoTime - t) / 1E9) + " seconds ")
    logger.info("Estimating non-singular factor parameters ...")
    val t1 = System.nanoTime
    val (fWeights,_) = fTrainer.trainWeights(factorGraph.factors)
    logger.info("Estimation finished in " + ((System.nanoTime - t1) / 1E9) + " seconds ")
    val sm = new FactorModel(new CategoricalGLPPredictor(singletonNN, true), sWeights)
    val fm = new FactorModel(new CategoricalGLPPredictor(factorNN, true), fWeights)
    new TrainedFactorGraph(fm, sm, factorGraph, fgSettings.sgAlpha)
  }  
}

class FactorGraph(val factors: Vector[MultiFactor], val singletons: Vector[SingletonFactor], val alphabets: AlphabetSet)

class TrainedFactorGraph(val factorModel: FactorModel, val singletonModel: FactorModel, 
    _fs: Vector[MultiFactor], _ss: Vector[SingletonFactor], _a: AlphabetSet, init: Double = 0.1, 
    inferType: String = "star") 
extends FactorGraph(_fs, _ss, _a) {
  def this(fm: FactorModel, sm: FactorModel, fg: FactorGraph, lr: Double) = this(fm, sm, fg.factors, fg.singletons, fg.alphabets, lr)
  def this(fm: FactorModel, sm: FactorModel, fg: FactorGraph, lr: Double, inferType: String) = this(fm, sm, fg.factors, fg.singletons, fg.alphabets, lr, inferType)
  
  //val inference = new SubgradientInference(factorModel, singletonModel, init)
  val inference = inferType match {
    case "subgrad" => new SubgradientInference(factorModel, singletonModel, init)
    case "smoothgrad" => new SmoothedGradientInference(factorModel, singletonModel, init)
    case _ => new StarCoordinatedBlockMinimizationInference(factorModel, singletonModel, init)
  }
  //val inference = new SmoothedGradientInference(factorModel, singletonModel, init)
  
  def mapInfer(n: Int) = {
    inference.mapInfer(factors, singletons, n)
  }
  
  def getMap() = {
    singletons map {s => s.getMode(singletonModel, true)}
  }
  
  def getAccuracy = {
    val cor = singletons.foldLeft(0){case (ac,v) => if (v.label == v.getMode(singletonModel, true)) ac + 1 else ac}
    cor.toDouble / singletons.length
  }
  
  def renderMapOutput(ofile: String, la: Alphabet) = {
    val o = new java.io.PrintWriter(new java.io.File(ofile))
    val invLa = la.getInverseMapping
    singletons foreach {v =>
      o.print(v.getInput.getUniqueKey.getOrElse("?"))
      o.print(' ')
      o.print(invLa(v.getMode(singletonModel, true)))
      o.print('\n')
    }
    o.close
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
                } else alphabet.ofString(a)
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
  
  
  def getFeatureAlphabetSingleton(fstr: String) : (Alphabet, Alphabet) = {
    val ifile = new java.io.File(fstr)
    val fa = new StdAlphabet()
    val la = new StdAlphabet()
    val lines = scala.io.Source.fromFile(ifile).getLines
    lines foreach {l =>
      l.split(' ').toList match {
        case _ :: lab :: rest =>
          la.ofString(lab)
          FactorGraph.getLineInfo(rest, fa,false)
        case _ =>
      }
    }
    (fa, la)
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

  def gatherFactorGraph(fgSettings: FactorGraphSettings) : FactorGraph = {
    val singletonStr = fgSettings.singletonFile
    val factorStr    = fgSettings.factorFile
    val (sa,sla) = getFeatureAlphabetSingleton(singletonStr)
    val fa = getFeatureAlphabetFactor(factorStr)
    val cardinality = sla.getSize // # of labels
    val fla = new IdentityAlphabet(cardinality*cardinality, false, true)
    sa.ensureFixed
    fa.ensureFixed
    sla.ensureFixed
    fla.ensureFixed
    val singletonExtractor = new GMSingletonExtractor(sa, sla)
    val singletons = scala.io.Source.fromFile(new java.io.File(singletonStr)).getLines.toList map singletonExtractor.extractFeatures
    val factorExtractor = new GMFactorExtractor(fa, fla, cardinality,singletonExtractor.variableToSingletonFactors)
    val factors = scala.io.Source.fromFile(new java.io.File(factorStr)).getLines.toList map factorExtractor.extractFeatures
    
    new FactorGraph(factors.toVector, singletons.toVector, AlphabetSet(sa, fa, sla, fla))
  }
  
  def gatherFactorGraph(fgSettings: FactorGraphSettings, alphabets: AlphabetSet) = {
    val sla = alphabets.sla
    val sa = alphabets.sa
    val fla = alphabets.fla
    val fa = alphabets.fa
    val singletonStr = fgSettings.singletonTestFile.get
    val factorStr    = fgSettings.factorTestFile.get
    val singletonExtractor = new GMSingletonExtractor(sa, sla)
    val singletons = scala.io.Source.fromFile(new java.io.File(singletonStr)).getLines.toList map singletonExtractor.extractFeatures
    val factorExtractor = new GMFactorExtractor(fa, fla, sla.getSize,singletonExtractor.variableToSingletonFactors)
    val factors = scala.io.Source.fromFile(new java.io.File(factorStr)).getLines.toList map factorExtractor.extractFeatures
    new FactorGraph(factors.toVector, singletons.toVector, alphabets)
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
  val factorSize = 2
  
  val bases = Vector.tabulate(factorSize){i => math.pow(varOrder,i).toInt}
  val numConfigs = math.pow(varOrder, factorSize).toInt
  
  def getAlphabet = factorAlphabet
  
  def getNumberOfFeatures = factorAlphabet.getSize
  
  def indexToAssignment(ind: Int) : Array[Int] = {
    val ar = Array.fill(factorSize)(0)
    var i = factorSize - 1
    var nn = ind
    while (i >= 0) {
      val base = bases(i)
      ar(i) = nn / base
      nn -= (base * ar(i))
      i -= 1
    }
    ar
  }
  
  
  val indexAssignmentMap = Array.tabulate(numConfigs)(indexToAssignment) 
  
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
        val mf = new MultiFactor(indexAssignmentMap, varOrder, sgs, dVec, (v1+v2))
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
    val uniqueId = p.head
    val rest = p.tail
        
    val lb = uniqueId.split('-').toList // ensure not a factor
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
        val l_ind = math.max(singletonLa.ofString(label), 0)
        val lv = DenseVec.zeros(singletonLa.getSize)
        lv.update(l_ind,1.0f) // one-hot encoding
        val sgf = new StdGLPFactor(-1, dVec, lv, Some(uniqueId))
        val factor = new SingletonFactor(sgf, l_ind)
        variableToSingletonFactors += (v1 -> factor)
        factor
      
      case _ => throw new RuntimeException("Invalid input: " + s)
    }
  }
  
}