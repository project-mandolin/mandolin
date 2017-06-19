package org.mitre.mandolin.xg

import org.mitre.mandolin.config.{ GeneralLearnerSettings, ConfigGeneratedCommandOptions }
import org.mitre.mandolin.glp.MandolinMLPSettings
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
  
  val maxDepth = asIntOpt("mandolin.xg.max-depth").getOrElse(5)
  val rounds   = asIntOpt("mandolin.xg.rounds").getOrElse(20)
  val objective = asStrOpt("mandolin.xg.objective").getOrElse("binary:logistic")
  val scalePosWeight = asFloatOpt("mandolin.xg.scale-pos-weight").getOrElse(1.0f)
  val gamma = asFloatOpt("mandolin.xg.gamma").getOrElse(5.0f)
  val silent = !(this.asBoolean("mandolin.xg.verbose"))

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