package org.mitre.mandolin.glp


import org.mitre.mandolin.util.{ StdAlphabet, PrescaledAlphabet, RandomAlphabet, Alphabet, IdentityAlphabet, 
AlphabetWithUnitScaling, IdentityAlphabetWithUnitScaling, IOAssistant, AbstractPrintWriter}
import org.mitre.mandolin.optimize.Updater
import org.mitre.mandolin.transform.{ FeatureExtractor, FeatureImportance }
import org.mitre.mandolin.gm.{ Feature, NonUnitFeature }
import org.mitre.mandolin.util.LineParser
import com.typesafe.config.Config
import scala.reflect.ClassTag
import net.ceedubs.ficus.Ficus._

case class GLPComponentSet(
    ann: ANNetwork, 
    predictor: CategoricalGLPPredictor, 
    outputConstructor: GLPPosteriorOutputConstructor, 
    featureExtractor: FeatureExtractor[String,GLPFactor],
    labelAlphabet: Alphabet,
    dim: Int,
    npts: Int)

/**
 * General driver/processor class for GLP and linear models. Includes various utility methods
 * used by subclasses for reading in training/test data, determining GLP model topology from
 * a specification provided in the configuration file and setting/interpreting all of the
 * various configuration options.
 * @param appSettings GLPModelSettings that specify everything for training/decoding/etc.
 * @author wellner
 */
abstract class AbstractProcessor extends LineParser {

  def getLabelAlphabet(labFile: Option[String] = None, io: IOAssistant): Alphabet = {
    if (labFile.isDefined) {
      val labels = io.readLines(labFile.get)
      val la = new StdAlphabet
      labels foreach { l => la.ofString(l) }
      la.ensureFixed
      la
    } else new StdAlphabet
  }
  
  def mapSpecToList(conf: Map[String, Map[String, String]]) = {
    val layerNames = conf.keySet
    var prevName = ""
    val nextMap = layerNames.toSet.foldLeft(Map():Map[String,String]){case (ac,v) =>
      val cc = conf(v)
        try {
        val inLayer = cc("data")
        ac + (inLayer -> v)
        } catch {case _:Throwable =>
          prevName = v  // this is the name for the input layer (as it has no "data" field")
          ac}
      }      
    var building = true    
    val buf = new collection.mutable.ArrayBuffer[String]    
    buf append prevName // add input layer name first
    while (building) {
      val current = nextMap.get(prevName)
      current match {case Some(c) => buf append c; prevName = c case None => building = false}
      }
    buf.toList map {n => conf(n)} // back out as an ordered list      
  }

  def getGLPSpec(cs: Map[String, Map[String,String]], idim: Int, odim: Int) : IndexedSeq[LType] = {
    getGLPSpec(mapSpecToList(cs), idim, odim)  
  }
  
  def getGLPSpec(cs: List[Map[String, String]], idim: Int, odim: Int): IndexedSeq[LType] = {
    for (i <- 0 until cs.length) yield {
      val l = cs(i)
      val ltype = l("ltype")
      val dim =
        if (i == 0) idim else if (i == (cs.length - 1)) odim
        else {
          if (l.contains("dim"))
            l("dim").toInt
          else throw new RuntimeException("Expected 'dim' provided in layer specification for non-input layer " + i)
        }
      val dropOut = l.get("dropout-ratio") match { case Some(v) => v.toFloat case None => 0.0f }
      val l1Pen = l.get("l1-pen") match { case Some(v) => v.toFloat case None => 0.0f }
      val l2Pen = l.get("l2-pen") match { case Some(v) => v.toFloat case None => 0.0f }
      val mn = l.get("max-norm") match { case Some(v) => v.toFloat case None => 0.0f }
      val cval = l.get("margin-size") match { case Some(v) => v.toFloat case None => 1.0f }
      val rlen = l.get("ramp-width") match { case Some(v) => v.toFloat case None => 2.0f }
      val seqLen = l.get("seq-len") match {case Some(v) => v.toInt case None => 0}
      ltype match {
        case "Input"        => LType(InputLType, dim, dropOut)
        case "InputSparse"  => LType(SparseInputLType, dim, dropOut)
        case "TanH"         => LType(TanHLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Logistic"     => LType(LogisticLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Linear"       =>
          val lt = if (i >= cs.length - 1) LinearOutLType else LinearLType
          println("dim = " + dim)
          LType(lt, dim, dropOut, l1Pen, l2Pen, mn)
        case "LinearNoBias" => LType(LinearNoBiasLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "CrossEntropy" => LType(CrossEntropyLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "Relu"         => LType(ReluLType, dim, dropOut, l1Pen, l2Pen, mn)
        case "SoftMax"      => LType(SoftMaxLType, dim, 0.0f, l1Pen, l2Pen, mn)
        case "Hinge"        => LType(HingeLType(cval), dim, 0.0f, l1Pen, l2Pen, mn)
        case "ModHuber"     => LType(ModHuberLType, dim, 0.0f, l1Pen, l2Pen, mn)
        case "Ramp"         => LType(RampLType(rlen), dim, 0.0f, l1Pen, l2Pen, mn)
        case "TransLog"     => LType(TransLogLType, dim, 0.0f, l1Pen, l2Pen, mn)
        case "TLogistic"    => LType(TLogisticLType, dim, 0.0f, l1Pen, l2Pen, mn)
        case "NegSampledSoftMax" => 
          val ss    = l.get("sample-size") match {case Some(ss) => ss.toInt case None => 5}
          val inDim = l.get("input-dim") match {case Some(id) => id.toInt case None => -1}
          val frequencyFile = l.get("frequency-file")
          LType(NegSampledSoftMaxLType(inDim, ss,frequencyFile.getOrElse("")), dim, 0.0f, l1Pen, l2Pen, mn)
        case a              => throw new RuntimeException("Unrecognized layer type: " + a)
      }
    }
  }

  def getAlphabet(inputLines: Vector[String], la: StdAlphabet, scaling: Boolean, selectFeatures: Int, fd: Option[String], io: IOAssistant): (Alphabet, Int) = {
    getAlphabet(inputLines.iterator, la, scaling, selectFeatures, fd, io)
  }
  
  def getScaledDenseVecAlphabet(inputLines: Iterator[String], la: Alphabet, d: Int) = {
    val alphabet = new IdentityAlphabetWithUnitScaling(d)
    var numPoints = 0
    inputLines foreach { l =>
      numPoints += 1
      val (y, _, _) = sparseOfLine(l, alphabet, buildVecs = false)
      la.ofString(y)
      }
    alphabet
  }

  def getAlphabet(inputLines: Iterator[String], la: Alphabet, scaling: Boolean, selectFeatures: Int, fdetailFile: Option[String], io: IOAssistant): (Alphabet, Int) = {
    val alphabet = if (scaling) new AlphabetWithUnitScaling else new StdAlphabet
    var numPoints = 0
    val fbuf = new collection.mutable.ArrayBuffer[Array[Feature]]
    val yvs = new collection.mutable.ArrayBuffer[Int]
    inputLines foreach { l =>
      numPoints += 1
      if (selectFeatures > 0) { // selecting features, so need to instantiate at this point...
        val (y, fv, _) = sparseOfLine(l, alphabet, buildVecs = true) // filter out features with a prefix
        fbuf append fv
        val yi = la.ofString(y)
        yvs append yi
      } else {
        val (y, _, _) = sparseOfLine(l, alphabet, buildVecs = false) // filter out features with a prefix
        la.ofString(y)
      }
    }
    val finalAlphabet =
      if ((selectFeatures > 0) && (selectFeatures < alphabet.getSize)) {
        val featureImpCompute = new FeatureImportance // basic feature selection
        val dataArr = fbuf.toArray
        val yvArr = yvs.toArray
        val numFeatures = alphabet.getSize
        val toKeep = featureImpCompute.computeMutualInfo(dataArr, yvArr, numFeatures, la.getSize, selectFeatures)
        val revMap = alphabet.mapping.foldLeft(Map[Int, String]()) { case (ac, (s, i)) => ac.updated(i, s) }         
        val newAlpha = new StdAlphabet
        toKeep foreach { i => newAlpha.ofString(revMap(i)) } // update newAlphabet only with features to keep
        val finalAlphabet = 
          if (scaling) {
            // if we're scaling features, we need to get the min and max values from original 
            // alphabet and map those into the reduced alphabet with selected features
            val origAlpha = alphabet.asInstanceOf[AlphabetWithUnitScaling]
            val maxVals = Array.fill(toKeep.length)(0.0)
            val minVals = Array.fill(toKeep.length)(0.0)
            toKeep foreach {i =>              
              val ni = newAlpha.ofString(revMap(i))
              maxVals(ni) = origAlpha.fmax(i)
              minVals(ni) = origAlpha.fmin(i)
              }            
            val psAlpha = new PrescaledAlphabet(minVals, maxVals)
            toKeep foreach {i => psAlpha.ofString(revMap(i))}
            psAlpha
          } else newAlpha
        finalAlphabet
      } else alphabet
    fdetailFile foreach { f =>
      val wr = io.getPrintWriterFor(f, false)
      finalAlphabet.mapping foreach {
        case (s, i) =>
          wr.write(s)
          wr.write('\n')
      }
      wr.close()
    }
    println("Feature symbol table extracted ... " + finalAlphabet.getSize + " features identified")
    if (selectFeatures > 0) {
      println((alphabet.getSize - finalAlphabet.getSize).toString + " features removed based on mutual information selection..\n")
    }
    (finalAlphabet,numPoints)
  }

  def getAlphabet(appSettings: GLPModelSettings, la: Alphabet, io: IOAssistant): (Alphabet, Int) = {
    if (appSettings.useRandom) (new RandomAlphabet(appSettings.numFeatures), 1000)
    else {
      println("Building alphabet with training input data")
      getAlphabet(io.readLines(appSettings.trainFile.get), la, appSettings.scaleInputs,
        appSettings.filterFeaturesMI, appSettings.printFeatureFile, io)
    }
  }
  
  def getSubComponents(appSettings: GLPModelSettings, idim: Int, odim: Int) : (ANNetwork, CategoricalGLPPredictor, GLPPosteriorOutputConstructor) = {
    val specs = appSettings.netspecConfig match {case Some(c) => getGLPSpec(c, idim, odim) case None => getGLPSpec(appSettings.netspec, idim, odim)}
    getSubComponents(specs)
  }

  def getSubComponents(confSpecs: List[Map[String, String]], idim: Int, odim: Int): (ANNetwork, CategoricalGLPPredictor, GLPPosteriorOutputConstructor) = {
    val specs = getGLPSpec(confSpecs, idim, odim)
    getSubComponents(specs)
  }
  
  def getSubComponents(mspec: IndexedSeq[LType], idim: Int, odim: Int): (ANNetwork, CategoricalGLPPredictor, GLPPosteriorOutputConstructor) = {
    val specs = ANNetwork.fullySpecifySpec(mspec, idim, odim)
    getSubComponents(specs)
  }

  def getSubComponents(specs: IndexedSeq[LType]): (ANNetwork, CategoricalGLPPredictor, GLPPosteriorOutputConstructor) = {
    val nn = ANNetwork(specs)
    val predictor = new CategoricalGLPPredictor(nn, true)
    val oc = new GLPPosteriorOutputConstructor
    (nn, predictor, oc)
  }

  def getComponentsDenseVecs(confSpecs: List[Map[String, String]], dim: Int, labelAlphabet: Alphabet, fa: Alphabet): GLPComponentSet = {    
    val isSparse = confSpecs.head("ltype").equals("InputSparse")
    val regression = confSpecs.last("ltype").equals("Linear")
    val fe =
      if (regression) new StdVectorExtractorRegression(fa, dim)
    else if (isSparse) new SparseVecFeatureExtractor(fa, labelAlphabet) else new StdVectorExtractorWithAlphabet(labelAlphabet, fa, dim)
    val laSize = if (regression) 1 else labelAlphabet.getSize
    val (nn, predictor, outConstructor) = getSubComponents(confSpecs, dim, laSize)
    GLPComponentSet(nn, predictor, outConstructor, fe, new IdentityAlphabet(1), dim, 1000)
  }

  def getComponentsDenseVecs(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    val fa = 
      if (appSettings.scaleInputs) getScaledDenseVecAlphabet(io.readLines(appSettings.trainFile.get), la, appSettings.denseVectorSize) 
      else new IdentityAlphabet(appSettings.denseVectorSize, fix = true)
    val cspec = appSettings.netspecConfig match {case Some(c) => mapSpecToList(c) case None => appSettings.netspec}  
    getComponentsDenseVecs(cspec, appSettings.denseVectorSize, la, fa)
  }  

  def getComponentsDenseVecs(layerSpecs: IndexedSeq[LType]): GLPComponentSet = {
    val (nn, predictor, outConstructor) = getSubComponents(layerSpecs)
    val labelAlphabet = new IdentityAlphabet(layerSpecs.last.dim)
    val fa = new IdentityAlphabet(layerSpecs.head.dim)
    val fe = new StdVectorExtractorWithAlphabet(labelAlphabet, fa, layerSpecs.head.dim)
    GLPComponentSet(nn, predictor, outConstructor, fe, labelAlphabet, layerSpecs.head.dim,1000)
  }

  def getComponentsHashsedFeatures(confSpecs: List[Map[String, String]], randFeatures: Int, labelAlphabet: StdAlphabet): GLPComponentSet = {
    val fa = new RandomAlphabet(randFeatures)
    val (nn, predictor, outConstructor) = getSubComponents(confSpecs, randFeatures, labelAlphabet.getSize)
    val fe = new SparseVecFeatureExtractor(fa, labelAlphabet)
    GLPComponentSet(nn, predictor, outConstructor, fe, labelAlphabet, randFeatures, 1000)
  }

  def getComponentsHashedFeatures(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    val fa = new RandomAlphabet(appSettings.numFeatures)
    val cspec = appSettings.netspecConfig match {case Some(c) => mapSpecToList(c) case None => appSettings.netspec}
    val (nn, predictor, outConstructor) = getSubComponents(cspec, appSettings.numFeatures, la.getSize)
    val fe = new SparseVecFeatureExtractor(fa, la)
    GLPComponentSet(nn, predictor, outConstructor, fe, la, appSettings.numFeatures,1000)
  }
  
  def getComponentsInducedAlphabet(mspec: IndexedSeq[LType], lines: Iterator[String],
                                   la: Alphabet, scale: Boolean = false, selectedFeatures: Int = -1, io: IOAssistant, faO: Option[(Alphabet,Int)] = None): GLPComponentSet = {    
    val (fa,npts) = faO match {
      case None => getAlphabet(lines, la, scale, selectedFeatures, None, io)
      case Some(x) => x // don't recompute alphabet if we've done so already
    }
    fa.ensureFixed
    la.ensureFixed
    val (nn, predictor, outConstructor) = getSubComponents(mspec, fa.getSize, la.getSize)
    val isSparse = ((mspec.length > 1) && (mspec(0).designate match {case SparseInputLType => true case _ => false}))
    val fe = if (isSparse) new SparseVecFeatureExtractor(fa, la) else new VecFeatureExtractor(fa, la)
    GLPComponentSet(nn, predictor, outConstructor, fe, la, fa.getSize, npts)
  }

  def getComponentsInducedAlphabet(appSettings: GLPModelSettings, lines: Iterator[String],
                                   la: Alphabet, scale: Boolean, selectedFeatures: Int, io: IOAssistant): GLPComponentSet = {
    val (fa,npts) = getAlphabet(lines, la, scale, selectedFeatures, None, io)
    fa.ensureFixed
    val mspec = appSettings.netspecConfig match {case Some(c) => getGLPSpec(c, fa.getSize, la.getSize) case None => getGLPSpec(appSettings.netspec, fa.getSize, la.getSize)}
    getComponentsInducedAlphabet(mspec, lines, la, scale, selectedFeatures, io, Some((fa,npts)))
  }

  def getComponentsInducedAlphabet(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    val la = getLabelAlphabet(appSettings.labelFile, io)
    val lines = io.readLines(appSettings.trainFile.get)
    getComponentsInducedAlphabet(appSettings, lines, la, appSettings.scaleInputs, appSettings.filterFeaturesMI, io)
  }
  
  def getComponentsViaSettings(appSettings: GLPModelSettings, io: IOAssistant): GLPComponentSet = {
    if (appSettings.denseVectorSize > 0) getComponentsDenseVecs(appSettings, io)
    else if (appSettings.useRandom) getComponentsHashedFeatures(appSettings, io)    
    else getComponentsInducedAlphabet(appSettings, io)
  }    

  def writeOutputs(os: AbstractPrintWriter, outputs: Iterator[(String, GLPFactor)], laOpt: Option[Alphabet]): Unit = {
    laOpt foreach { la =>
      if (la.getSize < 2) {
        os.print("ID,response,value\n")
        outputs foreach {case (s, factor) => os.print(s); os.print(','); os.print(factor.getOutput(0).toString); os.println}
      } else {
      val labelHeader = la.getMapping.toSeq.sortWith((a, b) => a._2 < b._2).map(_._1) // get label strings sorted by their index
      os.print("ID")
      for (i <- 0 until labelHeader.length) {
        os.print(',')
        os.print(labelHeader(i))
      }
      os.print(',')
      os.print("Label")
      os.println
      outputs foreach { case (s, factor) => os.print(s); os.print(','); os.print(factor.getOneHot.toString); os.println }
      }
    }
  }
}
