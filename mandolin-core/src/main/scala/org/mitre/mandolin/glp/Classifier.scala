package org.mitre.mandolin.glp

/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import org.mitre.mandolin.config._
import org.mitre.mandolin.util.{Alphabet, IOAssistant}
import org.mitre.mandolin.predict.OutputConstructor
import com.typesafe.config.Config
import net.ceedubs.ficus.Ficus._

class MandolinMLPSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends AppSettings[MandolinMLPSettings](_confOptions, _conf) with Serializable {

  val decoderInputFile = asStrOpt("mandolin.decoder.input-file")
  val outputFile = asStrOpt("mandolin.decoder.output-file")
  val inputModelFile = asStrOpt("mandolin.decoder.model-file")

  val numFeatures = asInt("mandolin.mmlp.num-hash-features")
  val trainFile = asStrOpt("mandolin.mmlp.train-file")
  val testFile = asStrOpt("mandolin.mmlp.test-file")
  val testFreq = asInt("mandolin.mmlp.eval-freq")
  val testPartitions = asInt("mandolin.mmlp.test-partitions")
  val modelFile = asStrOpt("mandolin.mmlp.model-file")

  val numEpochs = asInt("mandolin.mmlp.num-epochs")
  val numSubEpochs = asInt("mandolin.mmlp.num-subepochs")
  val detailsFile = asStrOpt("mandolin.mmlp.detail-file")
  val progressFile = asStrOpt("mandolin.mmlp.progress-file")
  val labelFile = asStrOpt("mandolin.mmlp.label-file")
  // val ensureSparse = asBoolean("mandolin.mmlp.ensure-sparse")
  val useRandom = asBoolean("mandolin.mmlp.use-random-features")
  val printFeatureFile = asStrOpt("mandolin.mmlp.print-feature-file")
  val filterFeaturesMI = asInt("mandolin.mmlp.max-features-mi")

  // these should move to model specification as they are specific to loss functions
  val coef1 = asDouble("mandolin.mmlp.coef1")
  val qval = asDouble("mandolin.mmlp.qval")

  val oversampleRatio = asDouble("mandolin.mmlp.oversample")
  val denseVectorSize = asInt("mandolin.mmlp.dense-vector-size")
  val scaleInputs = asBoolean("mandolin.mmlp.scale-inputs")
  val composeStrategy = asStr("mandolin.mmlp.updater-compose-strategy")
  val denseOutputFile = asStrOpt("mandolin.mmlp.dense-output-file") // output vectors in dense format
  val numThreads = asInt("mandolin.mmlp.threads")
  val skipProb: Double = asFloat("mandolin.mmlp.skip-probability")
  val miniBatchSize = asInt("mandolin.mmlp.mini-batch-size")
  val synchronous = asBoolean("mandolin.mmlp.synchronous")

  val sgdLambda = asFloat("mandolin.mmlp.optimizer.lambda")
  val epsilon = asFloat("mandolin.mmlp.optimizer.epsilon")
  val rho = asFloat("mandolin.mmlp.optimizer.rho")
  val method = asStr("mandolin.mmlp.optimizer.method")
  val initialLearnRate = asFloat("mandolin.mmlp.optimizer.initial-learning-rate")
  val maxNorm = asBoolean("mandolin.mmlp.optimizer.max-norm")


  def this(str: String) = this(None, Some(com.typesafe.config.ConfigFactory.parseString(str)))

  def this(args: Array[String]) = this(Some(new ConfigGeneratedCommandOptions(args)), None)

  def this() = this(Array(): Array[String])

  import scala.collection.JavaConversions._

  /**
    * Returns a new settings object with the config name `key` set to `v`
    */
  def set(key: String, v: Any) = {
    val curConfig = this.config
    val nConf = curConfig.withValue(key, com.typesafe.config.ConfigValueFactory.fromAnyRef(v))
    new MandolinMLPSettings(None, Some(nConf))
  }

  /**
    * Returns a new settings object with the sequence of tuple arguments values set accordingly
    */
  override def withSets(avs: Seq[(String, Any)]): MandolinMLPSettings = {
    val nc = avs.foldLeft(this.config) { case (ac, (v1, v2)) =>
      v2 match {
        case v2: List[_] =>
          if (v2 != null) ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromIterable(v2)) else ac
        case v2: Any =>
          ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromAnyRef(v2))
      }
    }
    new MandolinMLPSettings(None, Some(nc))
  }

  def mapSpecToList(conf: Map[String, Map[String, String]]) = {
    val layerNames = conf.keySet
    var prevName = ""
    val nextMap = layerNames.toSet.foldLeft(Map(): Map[String, String]) { case (ac, v) =>
      val cc = conf(v)
      try {
        val inLayer = cc("data")
        ac + (inLayer -> v)
      } catch {
        case _: Throwable =>
          prevName = v // this is the name for the input layer (as it has no "data" field")
          ac
      }
    }
    var building = true
    val buf = new collection.mutable.ArrayBuffer[String]
    buf append prevName // add input layer name first
    while (building) {
      val current = nextMap.get(prevName)
      current match {
        case Some(c) => buf append c; prevName = c
        case None => building = false
      }
    }
    buf.toList map { n => conf(n) } // back out as an ordered list
  }

  val netspec = try {
    config.as[List[Map[String, String]]]("mandolin.mmlp.specification")
  } catch {
    case _: Throwable =>
      Nil
  }
  val netspecConfig: Option[Map[String, Map[String, String]]] =
    try {
      Some(config.as[Map[String, Map[String, String]]]("mandolin.mmlp.specification"))
    }
    catch {
      case _: Throwable => None
    }
}

/**
  * Output constructor for GLP models. Writes a vector of outputs
  * which may form a distribution, e.g. with a SoftMax output layer.
  * Also includes comments/metadata from corresponding inputs so outputs
  * can be linked back to the data from which they came.
  *
  * @author wellner
  */
class GLPPosteriorOutputConstructor extends OutputConstructor[String, Seq[(Float, Int)], String] {
  def constructOutput(i: String, r: Seq[(Float, Int)], s: String): String = {
    val (vec, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: Seq[(Float, Int)]): String = {
    val sortedByY = r.sortWith((a, b) => a._2 < b._2)
    val sbuf = new StringBuilder
    sortedByY foreach {
      case (d, i) =>
        sbuf append ','
        sbuf append d.toString
    }
    sbuf.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}

class GLPMostProbableOutputConstructor extends OutputConstructor[String, Int, String] {
  def constructOutput(i: String, r: Int, s: String): String = {
    val (_, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: Int): String = {
    r.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}

class GLPRegressionOutputConstructor extends OutputConstructor[String, (Double, Double), String] {
  def constructOutput(i: String, r: (Double, Double), s: String): String = {
    val (vec, id) = i.split('#').toList match {
      case a :: b :: Nil => (a, Some(b))
      case a :: _ => (a, None)
      case Nil => throw new RuntimeException("Invalid line: " + i)
    }
    val idStr = id.getOrElse("-1")
    val resStr = responseString(r)
    val sbuf = new StringBuilder
    sbuf append idStr
    sbuf append resStr
    sbuf.toString
  }

  def responseString(r: (Double, Double)): String = {
    r._1.toString + ", " + r._2.toString
  }

  def intToResponseString(r: Int): String = "UNK"
}


/**
  * A separate utility program that takes inputs in a Sparse vector representation
  * and maps them to a tab-separate file dense representation with the feature
  * names (in the sparse vector input file) as column headers. The first column is
  * the label, which is the original string representation of the label in the
  * sparse input file.
  *
  * @author wellner
  */
object DenseVectorWriter {

  def exportVectorsDense(f: String, io: IOAssistant, fvs: Vector[GLPFactor], alphabet: Alphabet, la: Alphabet) = {
    val os = io.getPrintWriterFor(f, false)
    val invMap = la.getInverseMapping
    val im = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val ar = im.toArray
    os.write("key")
    os.write('\t')
    os.write("label")
    for (i <- 0 until ar.length) {
      os.write('\t')
      os.write(ar(i)._2.toString)
    }
    os.write('\n')
    fvs foreach { l =>
      val dv = l.getInput
      val outInd = l.getOneHot
      val outLab = invMap(outInd)
      l.getUniqueKey foreach { k =>
        os.write(k)
        os.write('\t')
      }
      os.write(outLab)
      for (i <- 0 until dv.getSize) {
        os.write('\t')
        os.write(dv(i).toString)
      }
      os.write('\n')
    }
    os.close()
  }

  def exportVectorsDense(f: java.io.File, fvs: Vector[GLPFactor], alphabet: Alphabet, la: Alphabet) = {
    val os = new java.io.PrintWriter(f)
    val invMap = la.getInverseMapping
    val im = alphabet.getInverseMapping.toList.sortWith { case ((k1, v1), (k2, v2)) => k1 < k2 }
    val ar = im.toArray
    os.write("key")
    os.write('\t')
    os.write("label")
    for (i <- 0 until ar.length) {
      os.write('\t')
      os.write(ar(i)._2.toString)
    }
    os.write('\n')
    fvs foreach { l =>
      val dv = l.getInput
      val outInd = l.getOneHot
      val outLab = invMap(outInd)
      l.getUniqueKey foreach { k =>
        os.write(k)
        os.write('\t')
      }
      os.write(outLab)
      for (i <- 0 until dv.getSize) {
        os.write('\t')
        os.write(dv(i).toString)
      }
      os.write('\n')
    }
    os.close()
  }
}
