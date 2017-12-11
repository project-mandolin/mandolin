package org.mitre.mandolin.mlp

import com.typesafe.config.Config
import org.mitre.mandolin.config.{AppSettings, ConfigGeneratedCommandOptions}

import net.ceedubs.ficus.Ficus._

/**
  * Created by jkraunelis on 6/21/17.
  */
class MandolinMLPSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) extends AppSettings[MandolinMLPSettings](_confOptions, _conf) with Serializable {

  val decoderInputFile = asStrOpt("mandolin.mmlp.test-file")
  val outputFile = asStrOpt("mandolin.mmlp.prediction-file")
  val inputModelFile = asStrOpt("mandolin.mmlp.model-file")
  val regression     = asBooleanOpt("mandolin.mmlp.regression").getOrElse(false)

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
