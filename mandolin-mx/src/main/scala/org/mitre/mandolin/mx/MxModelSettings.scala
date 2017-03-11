package org.mitre.mandolin.mx

//import org.mitre.mandolin.config.{LearnerSettings, OnlineLearnerSettings}
import org.mitre.mandolin.glp.{GLPModelSettings}
import org.mitre.mandolin.config.GeneralLearnerSettings
import com.typesafe.config.{Config}
import org.slf4j.LoggerFactory
import net.ceedubs.ficus.Ficus._

class MxModelSettings(a: Seq[String]) extends GLPModelSettings(a.toArray) {
  def this() = this(Seq())
  import scala.collection.JavaConversions._
  
  // input type: 1) glp, 2) ndarray, 3) recordio ... others?
  val inputType       = asStrOpt("mandolin.mx.input-type").getOrElse("glp")
  val numberOfClasses = asIntOpt("mandolin.mx.num-classes").getOrElse(2)
  
  //val specification = config.
  //val mxSpecification = try config.as[List[Config]]("mandolin.mx.specification") catch {case _: Throwable => Nil}
  val mxSpecification = try config.getAnyRefList("mandolin.mx.specification") catch {case _: Throwable => null}
  val gpus = try config.getAnyRefList("mandolin.mx.gpus") catch {case _: Throwable => null}
  val cpus = try config.getAnyRefList("mandolin.mx.cpus") catch {case _: Throwable => null}
  def getGpus = try config.as[List[Int]]("mandolin.mx.gpus") catch {case _: Throwable => Nil}
  def getCpus = try config.as[List[Int]]("mandolin.mx.cpus") catch {case _: Throwable => Nil}
  val saveFreq = asIntOpt("mandolin.mx.save-freq").getOrElse(10)
  
  // this should override mandolin global learning rate  
  val mxInitialLearnRate = asFloatOpt("mandolin.mx.train.initial-learning-rate").getOrElse(0.01f)
  val mxRescaleGrad       = asFloatOpt("mandolin.mx.train.rescale-gradient").getOrElse(0f)
  val mxMomentum         = asFloatOpt("mandolin.mx.train.momentum").getOrElse(0f)
  val mxGradClip         = asFloatOpt("mandolin.mx.train.gradient-clip").getOrElse(0f)
  val mxRho              = asFloatOpt("mandoiin.mx.train.rho").getOrElse(0.01f)
  
  // values are: sgd, adam, rmsprop, adadelta, nag, adagrad, sgld
  val mxOptimizer        = asStrOpt("mandolin.mx.train.optimizer").getOrElse("sgd")
 
  val channels = asIntOpt("mandolin.mx.img.channels").getOrElse(0)
  val xdim     = asIntOpt("mandolin.mx.img.xdim").getOrElse(0)
  val ydim     = asIntOpt("mandolin.mx.img.ydim").getOrElse(0)
  val meanImgFile = asStrOpt("mandolin.mx.img.mean-image").getOrElse("mean-img")
  val preProcThreads = asIntOpt("mandolin.mx.img.preprocess-threads").getOrElse(8)
  
  override def withSets(avs: Seq[(String, Any)]) : MxModelSettings  = {
    val nc = avs.foldLeft(this.config){case (ac, (v1,v2)) => 
      v2 match {
        case v2: List[_] =>
          if (v2 != null) ac.withValue(v1,com.typesafe.config.ConfigValueFactory.fromIterable(v2)) else ac
        case v2: Any => ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromAnyRef(v2))}
      }
      
    new MxModelSettings() {
      override lazy val config = nc
    }
  }
}