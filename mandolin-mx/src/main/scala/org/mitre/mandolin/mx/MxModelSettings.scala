package org.mitre.mandolin.mx

import org.mitre.mandolin.config.ConfigGeneratedCommandOptions
import org.mitre.mandolin.mlp.MandolinMLPSettings
import com.typesafe.config.Config
import net.ceedubs.ficus.Ficus._

class MxModelSettings(_confOptions: Option[ConfigGeneratedCommandOptions], _conf: Option[Config]) 
extends MandolinMLPSettings(_confOptions, _conf) {
  
  def this(s: String) = this(None,Some(com.typesafe.config.ConfigFactory.parseString(s)))
  def this(args: Seq[String]) = this(Some(new ConfigGeneratedCommandOptions(args)),None)
  def this() = this(Seq())
  
  import scala.collection.JavaConversions._
  
  // input type: 1) mlp, 2) ndarray, 3) recordio ... others?
  val inputType       = asStrOpt("mandolin.mx.input-type").getOrElse("mlp")
  val numberOfClasses = asIntOpt("mandolin.mx.num-classes").getOrElse(2)

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
  val mxRho              = asFloatOpt("mandolin.mx.train.rho").getOrElse(0.01f)
  val mxWd               = asFloatOpt("mandolin.mx.train.wd").getOrElse(0.00001f) // L2 weight decay
  val mxInitializer      = asStrOpt("mandolin.mx.train.initializer").getOrElse("uniform")
  
  val mxTrainLabels      = asStrOpt("mandolin.mx.train-labels")
  val mxTestLabels      = asStrOpt("mandolin.mx.test-labels")
  
  
  // values are: sgd, adam, rmsprop, adadelta, nag, adagrad, sgld
  val mxOptimizer        = asStrOpt("mandolin.mx.train.optimizer").getOrElse("sgd")
 
  val channels = asIntOpt("mandolin.mx.img.channels").getOrElse(0)
  val xdim     = asIntOpt("mandolin.mx.img.xdim").getOrElse(0)
  val ydim     = asIntOpt("mandolin.mx.img.ydim").getOrElse(0)
  val meanImgFile = asStrOpt("mandolin.mx.img.mean-image").getOrElse("mean-img")
  val preProcThreads = asIntOpt("mandolin.mx.img.preprocess-threads").getOrElse(8)
  val mxResizeShortest   = asIntOpt("mandolin.mx.img.resize-shortest").getOrElse(0)
  
  // this allows GPU hosts to be specified in the configuration
  val gpuHosts = try config.as[List[String]]("mandolin.mx.gpu-hosts") catch {case _:Throwable => Nil}
  
  // set this up to have a device mapping
  // gpu-host1 => 0,1,2,3, gpu-host2 => 0,1, etc.
  val gpuHostMapping = try config.getAnyRefList("mandolin.mx.gpu-host-map") catch {case _:Throwable => null}
  
  override def withSets(avs: Seq[(String, Any)]) : MxModelSettings  = {
    val nc = avs.foldLeft(this.config){case (ac, (v1,v2)) =>       
      v2 match {
        case v2: List[_] =>
          if (v2 != null) ac.withValue(v1,com.typesafe.config.ConfigValueFactory.fromIterable(v2)) else ac
        case v2: Any =>
          ac.withValue(v1, com.typesafe.config.ConfigValueFactory.fromAnyRef(v2))}
      }
    new MxModelSettings(None,Some(nc)) 
  }
}