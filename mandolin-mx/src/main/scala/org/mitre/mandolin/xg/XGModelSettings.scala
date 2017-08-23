package org.mitre.mandolin.xg

import org.mitre.mandolin.config.ConfigGeneratedCommandOptions
import org.mitre.mandolin.mlp.MandolinMLPSettings
import com.typesafe.config.Config

/**
 * @author wellner
 */
class XGModelSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) 
extends MandolinMLPSettings(_confOptions, _conf) {
  import scala.collection.JavaConversions._
  
  def this(s: String) = this(None,Some(com.typesafe.config.ConfigFactory.parseString(s)))
  def this(args: Seq[String]) = this(Some(new ConfigGeneratedCommandOptions(args)),None)
  def this() = this(Seq())
  
  override val trainFile = asStrOpt("mandolin.xg.train-file")
  override val testFile = asStrOpt("mandolin.xg.test-file")
  override val labelFile = asStrOpt("mandolin.xg.label-file")
  override val modelFile = asStrOpt("mandolin.xg.model-file")
  override val outputFile = asStrOpt("mandolin.xg.prediction-file")
  override val numThreads = asIntOpt("mandolin.xg.threads").getOrElse(1)
  override val denseVectorSize = asIntOpt("mandolin.xg.dense-vector-size").getOrElse(0)

  val maxDepth = asIntOpt("mandolin.xg.max-depth").getOrElse(5)
  val rounds   = asIntOpt("mandolin.xg.rounds").getOrElse(20)
  val objective = asStrOpt("mandolin.xg.objective").getOrElse("binary:logistic")
  val scalePosWeight = asFloatOpt("mandolin.xg.scale-pos-weight").getOrElse(1.0f)
  val gamma = asFloatOpt("mandolin.xg.gamma").getOrElse(5.0f)
  val silent = !(this.asBoolean("mandolin.xg.verbose"))
  val evalMethod = asStrOpt("mandolin.xg.eval-method") match {
    case Some(m) => m
    case None => if (objective.equals("binary:logistic")) "error" else "merror"
  }
  

  /**
   * Returns a new settings object with the sequence of tuple arguments values set accordingly
   */
  override def withSets(avs: Seq[(String, Any)]) : XGModelSettings  = {
    val nc = avs.foldLeft(this.config){case (ac, (v1,v2)) => 
      v2 match {
        case v2: List[_] =>
          if (v2 != null) ac.withValue(v1,com.typesafe.config.ConfigValueFactory.fromIterable(v2)) else ac
        case v2: Any =>
          ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromAnyRef(v2))}
      }    
    new XGModelSettings(None,Some(nc))     
  }
}