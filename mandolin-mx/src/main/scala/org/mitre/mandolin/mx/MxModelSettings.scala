package org.mitre.mandolin.mx

//import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings}
import org.mitre.mandolin.glp.GLPModelSettings
import com.typesafe.config.{Config}
import net.ceedubs.ficus.Ficus._

class MxModelSettings(a: Seq[String]) extends GLPModelSettings(a.toArray) {

  // input type: 1) glp, 2) ndarray, 3) recordio ... others?
  val inputType = asStrOpt("mandolin.mx.input-type")
  val numberOfClasses = asInt("mandolin.mx.num-classes")
  
  val gpus = config.as[List[Int]]("mandolin.mx.gpus")
  val cpus = config.as[List[Int]]("mandolin.mx.cpus")
  val saveFreq = asIntOpt("mandolin.mx.save-freq")
  
  // this should override mandolin global learning rate
  val mxInitialLearnRate = asFloatOpt("mandolin.mx.train.initial-learning-rate")
  val mxRescaleGrad       = asFloatOpt("mandolin.mx.train.rescale-gradient")
  val mxMomentum         = asFloatOpt("mandolin.mx.train.momentum")
  val mxGradClip         = asFloatOpt("mandolin.mx.train.gradient-clip")
  val mxRho              = asFloatOpt("mandoiin.mx.train.rho")
  
  // values are: sgd, adam, rmsprop, adadelta, nag, adagrad, sgld
  val mxOptimizer        = asStrOpt("mandolin.mx.train.optimizer")
 
  val channels = asIntOpt("mandolin.mx.img.channels")
  val xdim     = asIntOpt("mandolin.mx.img.xdim")
  val ydim     = asIntOpt("mandolin.mx.img.ydim")
  val meanImgFile = asStrOpt("mandolin.mx.img.mean-image")
  val preProcThreads = asIntOpt("mandolin.mx.img.preprocess-threads")
}