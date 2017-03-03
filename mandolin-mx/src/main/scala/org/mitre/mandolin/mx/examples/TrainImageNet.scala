package org.mitre.mandolin.mx

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.Callback._
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Initializer
import ml.dmlc.mxnet.Callback._

// import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

class ImageNetTrainer {

 private val logger = LoggerFactory.getLogger(classOf[ImageNetTrainer])

  def convFactory(data: Symbol, numFilter: Int, kernel: String, stride: String = "(1,1)",
                  pad: String = "(0,0)", name: scala.Option[String] = None) = {
    val conv = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> numFilter,
                         "kernel" -> kernel, "stride" -> stride, "pad" -> pad))
    val act = Symbol.Activation()()(Map("data" -> conv, "act_type" -> "relu"))
    act
  }
 
  def convFactoryBN(data: Symbol, numFilter: Int, kernel: String, stride: String = "(1,1)",
                  pad: String = "(0,0)", name: scala.Option[String] = None) = {
    val conv = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> numFilter,
                         "kernel" -> kernel, "stride" -> stride, "pad" -> pad))
    val bn = Symbol.BatchNorm()()(Map("data" -> conv))                         
    val act = Symbol.Activation()()(Map("data" -> bn, "act_type" -> "relu"))
    act
  }

  def inceptionFactory(data: Symbol, num1x1: Int, num3x3red: Int, num3x3: Int, numd5x5red: Int,
                       numd5x5: Int, pool: String, proj: Int, name: scala.Option[String] = None) = {
    val c1x1  = convFactory(data, num1x1, "(1,1)")
    val c3x3r = convFactory(data, num3x3red, "(1,1)")
    val c3x3  = convFactory(c3x3r, num3x3, "(3,3)", pad = "(1,1)")
    val cd5x5r = convFactory(data, numd5x5red, "(1,1)")
    val cd5x5 = convFactory(cd5x5r, numd5x5, "(5,5)", pad = "(2,2)")
    val pooling = Symbol.Pooling()()(Map("data" -> data, "kernel" -> "(3,3)", "stride" -> "(1,1)", "pad" -> "(1,1)",
                                   "pool_type" -> pool))
    val cproj = convFactory(pooling, proj, "(1,1)")
    val concat = Symbol.Concat()(c1x1,c3x3,cd5x5,cproj)()
    concat
  }
  
  def inceptionFactoryA(data: Symbol, num1x1: Int, num3x3red: Int, num3x3: Int, numd3x3red: Int,
                       numd3x3: Int, pool: String, proj: Int, name: scala.Option[String] = None) = {
    val c1x1  = convFactoryBN(data, num1x1, "(1,1)")
    val c3x3r = convFactoryBN(data, num3x3red, "(1,1)")
    val c3x3  = convFactoryBN(c3x3r, num3x3, "(3,3)", pad = "(1,1)")
    val cd3x3r = convFactoryBN(data, num3x3red, "(1,1)")
    val cd3x3_1 = convFactoryBN(cd3x3r, numd3x3, "(3,3)", pad = "(1,1)")
    val cd3x3_2 = convFactoryBN(cd3x3_1, numd3x3, "(3,3)", pad = "(1,1)")
    val pooling = Symbol.Pooling()()(Map("data" -> data, "kernel" -> "(3,3)", "stride" -> "(1,1)", "pad" -> "(1,1)",
                                   "pool_type" -> pool))
    val cproj = convFactoryBN(pooling, proj, "(1,1)")
    val concat = Symbol.Concat()(c1x1, c3x3, cd3x3_2, cproj)()
    concat
  }
  
  def inceptionFactoryB(data: Symbol, num3x3red: Int, num3x3: Int, numd3x3red: Int,
                       numd3x3: Int, name: scala.Option[String] = None) = {
    val c3x3r  = convFactoryBN(data, num3x3red, "(1,1)")
    val c3x3   = convFactoryBN(c3x3r, num3x3, "(3,3)", pad = "(1,1)", stride = "(2,2)")

    val cd3x3r = convFactoryBN(data, numd3x3red, "(1,1)")
    val cd3x3_1 = convFactoryBN(cd3x3r, numd3x3, "(3,3)", pad = "(1,1)")
    val cd3x3_2 = convFactoryBN(cd3x3_1, numd3x3, "(3,3)", pad = "(1,1)", stride = "(2,2)")
    val pooling = Symbol.Pooling()()(Map("data" -> data, "kernel" -> "(3,3)", "stride" -> "(2,2)", "pad" -> "(1,1)",
                                   "pool_type" -> "max"))
    val concat = Symbol.Concat()(c3x3, cd3x3_2, pooling)()
    concat
  }
  
  def getInceptionBN(numClasses: Int) : Symbol = {
    val data = Symbol.Variable("data")
    val conv1 = convFactoryBN(data, 64, "(7,7)", stride="(2,2)", pad="(3,3)")
    val pool1 = Symbol.Pooling()()(Map("data" -> conv1, "kernel" -> "(3,3)", "stride" -> "(2,2)", "pool_type" -> "max"))
    val conv2 = convFactoryBN(pool1, 64, "(1,1)", stride="(1,1)")
    val conv3 = convFactoryBN(conv2, 192, "(3,3)", stride="(1,1)", pad="(1,1)")
    val pool2 = Symbol.Pooling()()(Map("data" -> conv3, "kernel"->"(3,3)", "stride"->"(2,2)", "pool_type"->"max"))
        
    val in3a  = inceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32)
    val in3b  = inceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64)
    val in3c  = inceptionFactoryB(in3b, 128, 160, 64, 96)    
        
    val in4a  = inceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128)
    val in4b  = inceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128)
    val in4c  = inceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128)
    val in4d  = inceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128)
    val in4e  = inceptionFactoryB(in4d, 128, 192, 192, 256)
    
    val in5a = inceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128)
    val in5b = inceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128)
    
    val avg  = Symbol.Pooling()()(Map("data" -> in5b, "kernel"->"(7,7)", "stride" -> "(1,1)", "pool_type"->"avg"))
    val flatten = Symbol.Flatten()()(Map("data" -> avg))
    val fc1 = Symbol.FullyConnected()()(Map("data"->flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc1))
    softmax
  }

   def getGoogleNet(numClasses: Int) : Symbol = {
    val data = Symbol.Variable("data")
    val conv1 = convFactory(data, 64, "(7,7)", stride="(2,2)", pad="(3,3)")
    val pool1 = Symbol.Pooling()()(Map("data" -> conv1, "kernel" -> "(3,3)", "stride" -> "(2,2)", "pool_type"->"max"))
    val conv2 = convFactory(pool1, 64, "(1,1)", stride="(1,1)")
    val conv3 = convFactory(conv2, 192, "(3,3)", stride="(1,1)", pad="(1,1)")
    val pool3 = Symbol.Pooling()()(Map("data" -> conv3, "kernel"->"(3,3)", "stride"->"(2,2)", "pool_type"->"max"))

    val in3a  = inceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32)
    val in3b  = inceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64)
    val pool4 = Symbol.Pooling()()(Map("data" -> in3b, "kernel"->"(3,3)", "stride"->"(2,2)", "pool_type"->"max"))
    val in4a  = inceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64)
    val in4b  = inceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64)
    val in4c  = inceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64)
    val in4d  = inceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64)
    val in4e  = inceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128)
    val pool5 = Symbol.Pooling()()(Map("data" -> in4e, "kernel"->"(3,3)", "stride"->"(2,2)", "pool_type"->"max"))
    val in5a  = inceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128)
    val in5b  = inceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128)
    val pool6 = Symbol.Pooling()()(Map("data" -> in5b, "kernel" -> "(7,7)", "stride" -> "(1,1)", "pool_type"->"avg"))
    val flatten = Symbol.Flatten()()(Map("data" -> pool6))
    val fc1 = Symbol.FullyConnected()()(Map("data"->flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc1))
    softmax
  }
   
  def getVgg(numClasses: Int) : Symbol = {
    val data = Symbol.Variable("data")
    val conv1_1 = Symbol.Convolution()()(Map("data" -> data,
                                       "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 64))
    val relu1_1 = Symbol.Activation()()(Map("data" -> conv1_1, "act_type" -> "relu"))
    val pool1 = Symbol.Pooling()()(Map("data" -> relu1_1, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))
    // group 2
    val conv2_1 = Symbol.Convolution()()(Map("data" -> pool1, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 128))
    val relu2_1 = Symbol.Activation()()(Map("data" -> conv2_1, "act_type" -> "relu"))
    val pool2 = Symbol.Pooling()()(Map("data" -> relu2_1, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))

    // group 3
    val conv3_1 = Symbol.Convolution()()(Map("data" -> pool2, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 256))
    val relu3_1 = Symbol.Activation()()(Map("data" -> conv3_1, "act_type" -> "relu"))
    val conv3_2 = Symbol.Convolution()()(Map("data" -> relu3_1, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 256))
    val relu3_2 = Symbol.Activation()()(Map("data" -> conv3_2, "act_type" -> "relu"))
    val pool3 = Symbol.Pooling()()(Map("data" -> relu3_2, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))

    // group 4
    val conv4_1 = Symbol.Convolution()()(Map("data" -> pool3, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 512))
    val relu4_1 = Symbol.Activation()()(Map("data" -> conv4_1, "act_type" -> "relu"))
    val conv4_2 = Symbol.Convolution()()(Map("data" -> relu4_1, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 512))
    val relu4_2 = Symbol.Activation()()(Map("data" -> conv4_2, "act_type" -> "relu"))
    val pool4 = Symbol.Pooling()()(Map("data" -> relu4_2, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))

    // group 5
    val conv5_1 = Symbol.Convolution()()(Map("data" -> pool4, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 512))
    val relu5_1 = Symbol.Activation()()(Map("data" -> conv5_1, "act_type" -> "relu"))
    val conv5_2 = Symbol.Convolution()()(Map("data" -> relu5_1, "kernel" -> "(3,3)", "pad" -> "(1,1)", "num_filter" -> 512))
    val relu5_2 = Symbol.Activation()()(Map("data" -> conv5_2, "act_type" -> "relu"))
    val pool5 = Symbol.Pooling()()(Map("data" -> relu5_2, "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))

    // group 6
    val flatten = Symbol.Flatten()()(Map("data" -> pool5))
    val fc6 = Symbol.FullyConnected()()(Map("data" -> flatten, "num_hidden" -> 4096))
    val relu6 = Symbol.Activation()()(Map("data" -> fc6, "act_type" -> "relu"))
    val drop6 = Symbol.Dropout()()(Map("data" -> relu6, "p" -> 0.5))
    val fc7 = Symbol.FullyConnected()()(Map("data" -> drop6, "num_hidden" -> 4096))
    val relu7 = Symbol.Activation()()(Map("data" -> fc7, "act_type" -> "relu"))
    val drop7 = Symbol.Dropout()()(Map("data" -> relu7, "p" -> 0.5))

    val fc8 = Symbol.FullyConnected(name="fc8")()(Map("data" -> drop7, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc8))
    softmax
  }
  
  def getIterator(dataShape: Shape, trainStr: String, valStr: String)(dataDir: String, batchSize: Int, kv: KVStore) : (DataIter, DataIter) = {
    val train = IO.ImageRecordIter(Map(
      "path_imgrec" -> (dataDir + "/" + trainStr),
      "label_name" -> "softmax_label",
      // "mean_r" -> "123.68",
      // "mean_g" -> "116.779",
      // "mean_b" -> "103.939",
      "mean_img" -> (dataDir + "/" + "mean.bin"), // will generate this mean file if it doesn't exist from data
      "data_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "rand_crop" -> "True",
      "rand_mirror" -> "True",
      "shuffle" -> "True"
      )
    )

    val test = IO.ImageRecordIter(Map(
      "path_imgrec" -> (dataDir + "/" + trainStr),
      "label_name" -> "softmax_label",
      // "mean_r" -> "123.68",
      // "mean_g" -> "116.779",
      // "mean_b" -> "103.939",
      "mean_img" -> (dataDir + "/" + "mean.bin"), // will use this as mean if it exists
      "data_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "rand_crop" -> "False",
      "rand_mirror" -> "False"
      )
    )
    (train, test)
  }    

}

object TrainImageNet {

  val shape = 224 // size of edge, assumed square
  val dataShape = Shape(3, shape, shape)
  val tr = new ImageNetTrainer

  
  def main(args: Array[String]) : Unit = {
    val classes = args(0)
    val dataDir = args(1)
    val trPath = args(2)
    val tstPath = args(3)
    val deviceType = args(4)
    val idRange = args(5)
    val (st, en) = idRange.split('-').toList match {case s :: e :: _ => (s.toInt,e.toInt) case s :: _ => (s.toInt, s.toInt)}
    val ctxL = deviceType match {
      case "gpu" => for (i <- st to en) yield Context.gpu(i)
      case "cpu" => for (i <- st to en) yield Context.cpu(i)
      case _ => throw new RuntimeException("Invalid device type: " + deviceType) 
    }
    val batchSize = if (args.length > 6) args(6).toInt else 32
    val (trIter, tstIter) = tr.getIterator(dataShape, trPath, tstPath)(dataDir, batchSize, null)
    val net = tr.getInceptionBN(classes.toInt)
    // val net = tr.getGoogleNet(classes.toInt)
    val ctx = ctxL.toArray
    val scheduler = new FactorScheduler(trIter.size, 0.94f) // update by 0.94 after each epoch 
    val sgd = new SGD(learningRate = 0.015f, momentum = 0.9f, wd = 0.00001f, clipGradient = 4.0f, lrScheduler = scheduler)
    val updater = new MxNetOptimizer(sgd)

    val weights = new MxNetWeights(1.0f)
    val evaluator = new MxNetEvaluator(net, ctx, dataShape, batchSize, Some("model"))
    evaluator.evaluateTrainingMiniBatch(trIter, tstIter, weights, updater, 50)
    
    ()
  }
}