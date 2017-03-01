package org.mitre.mandolin.mx

import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings}
import com.typesafe.config.{Config}
import net.ceedubs.ficus.Ficus._

class MxModelSettings(a: Seq[String]) extends LearnerSettings(a) with OnlineLearnerSettings {

  // input type: 1) glp, 2) ndarray, 3) recordio ... others?
  val inputType = asStrOpt("mandolin.mx.input-type")
  val numberOfClasses = asInt("mandolin.mx.num-classes")
  
  val gpus = config.as[List[Int]]("mandolin.mx.gpus")
  val cpus = config.as[List[Int]]("mandolin.mx.cpus")
 
  val channels = asIntOpt("mandolin.mx.img.channels")
  val xdim     = asIntOpt("mandolin.mx.img.xdim")
  val ydim     = asIntOpt("mandolin.mx.img.ydim")
  val meanImgFile = asStrOpt("mandolin.mx.img.mean-image")
  val preProcThreads = asIntOpt("mandolin.mx.img.preprocess-threads")
}