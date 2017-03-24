package org.mitre.mandolin.mx

import ml.dmlc.mxnet._
import com.typesafe.config.{Config, ConfigValue}
import net.ceedubs.ficus.Ficus._


/**
 * Provides functionality for building up MXNet network/symbol objects from JSON-syntax
 * specification.
 * Structure, example:
 * 
 * {"type":"variable", "name": "input"}  // data refers to the name of the input for this symbol
 * {"type":"convBN", "name": "conv1", "data":"input", 
 *  "spec":{"numFilter": 64, "kernel":[1,1], "stride":[1,1], "pad":[0,0]}} 
 *
 *
 * {"type":"pooling", "name": "pool1", "data": "conv1",
 *  "spec":{"kernel":[3,3], "stride":[2,2], "pool_type": "max"} }
 * 
 * Could also handle composite definitions here by allowing for user-specified "types" 
 *
 */
class SymbolBuilder {
  
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
  
  
  private def getAsTuple2String(spec: Config, key: String, default: String = "(1,1)") : String = {
    try {
      val li = spec.as[List[String]](key)
      "("+li(0)+","+li(1)+")"
    } catch {case _: Throwable => default} 
  }
  
  /**
   * This constructs a "Pooling" symbol from the config specification
   */
  def mapFromPoolSpec(spec: Config, in: Symbol) : Map[String, AnyRef] = {    
    val kernel = getAsTuple2String(spec, "kernel")
    val stride = getAsTuple2String(spec, "stride")
    val pad    = getAsTuple2String(spec, "pad", "(0,0)")
    val poolType = spec.as[String]("pool_type")
    Map("data" -> in, "kernel" -> kernel, "stride" -> stride, "pad" -> pad, "pool_type" -> poolType)        
  }
  
  
  def convSymbolFromSpec(spec: Config, in: Symbol) : Symbol = {
    val nf = spec.as[Int]("num_filter")
    val kernel = getAsTuple2String(spec, "kernel")
    val stride = getAsTuple2String(spec, "stride")
    val pad    = getAsTuple2String(spec, "pad", "(0,0)")
    convFactory(in, nf, kernel, stride, pad)
  }
  
  def convBNSymbolFromSpec(spec: Config, in: Symbol) : Symbol = {
    val nf = spec.as[Int]("num_filter")
    val kernel = getAsTuple2String(spec, "kernel")
    val stride = getAsTuple2String(spec, "stride")
    val pad    = getAsTuple2String(spec, "pad", "(0,0)")
    convFactoryBN(in, nf, kernel, stride, pad)
  }
  
  def inceptionSymbolFromSpec(spec: Config, in: Symbol) : Symbol = {
    val num1x1 = spec.as[Int]("n1x1")
    val num3x3r = spec.as[Int]("n3x3r")
    val num3x3  = spec.as[Int]("n3x3")
    val num5x5r = spec.as[Int]("n5x5r")
    val num5x5  = spec.as[Int]("n5x5")
    val pt      = spec.as[String]("pool_type")
    val proj    = spec.as[Int]("projection")
    inceptionFactory(in, num1x1, num3x3r, num3x3, num5x5r, num5x5, pt, proj)
  }
  
  def inceptionASymbolFromSpec(spec: Config, in: Symbol) : Symbol = {
    val num1x1 = spec.as[Int]("n1x1")
    val num3x3r = spec.as[Int]("n3x3r")
    val num3x3  = spec.as[Int]("n3x3")
    val num5x5r = spec.as[Int]("n5x5r")
    val num5x5  = spec.as[Int]("n5x5")
    val pt      = spec.as[String]("pool_type")
    val proj    = spec.as[Int]("projection")
    inceptionFactoryA(in, num1x1, num3x3r, num3x3, num5x5r, num5x5, pt, proj)
  }
  
  def inceptionBSymbolFromSpec(spec: Config, in: Symbol) : Symbol = {
    val num3x3r = spec.as[Int]("n3x3r")
    val num3x3  = spec.as[Int]("n3x3")
    val numd3x3r  = spec.as[Int]("nd3x3r")
    val numd3x3  = spec.as[Int]("nd3x3")
    inceptionFactoryB(in, num3x3r, num3x3, numd3x3r, numd3x3)
  }
  
  def fullyConnectedFromSpec(spec: Config, in: Symbol) : Symbol = {
    val dim = spec.as[Int]("num_hidden")
    Symbol.FullyConnected()()(Map("data"-> in, "num_hidden" -> dim))
  }
  
  def flattenFromSpec(spec: Config, in: Symbol) : Symbol = {
    Symbol.Flatten()()(Map("data" -> in))
  }
  
  def activationFromSpec(spec: Config, in: Symbol) : Symbol = {
    Symbol.Activation()()(Map("data" -> in, "act_type" -> spec.as[String]("act_type")))    
  }
  
  def dropOutFromSpec(spec: Config, in: Symbol) : Symbol = {
    Symbol.Dropout()()(Map("data" -> in, "p" -> spec.as[String]("p")))
  }
  
  def softMaxFromSpec(spec: Config, in: Symbol) : Symbol = {
    Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> in))
  }
  
  def batchNormFromSpec(spec: Config, in: Symbol) : Symbol = {
    val eps = try spec.as[String]("eps") catch {case _: Throwable => "0.001"}
    val momentum = try spec.as[String]("momentum") catch {case _: Throwable => "0.9"}
    val fixGamma = try spec.as[String]("fix_gamma") catch {case _: Throwable => "True"}
    Symbol.BatchNorm()()(Map("data" -> in, "eps" -> eps, "momentum" -> momentum, "fix_gamma" -> fixGamma))
  }
  
  def resNetV2CoreFromSpec(spec: Config, in: Symbol) : Symbol = {
    val units = spec.as[List[Int]]("units")
    val filterList = spec.as[List[Int]]("filter_list")
    val bottleNeck = spec.as[Boolean]("bottle_neck")
    WideResNetSymbols.resnetV2(in, units, filterList, bottleNeck, 512)
  }
  
  def mxConvolutionFromSpec(spec: Config, in: Symbol) : Symbol = {
    val numFilters = spec.as[Int]("num_filter")
    val kernel = getAsTuple2String(spec, "kernel")
    val stride = getAsTuple2String(spec, "stride")
    val pad    = getAsTuple2String(spec, "pad", "(0,0)")
    val noBias = try spec.as[Boolean]("no_bias") catch {case _:Throwable => false}
    val workSpace = try spec.as[Int]("workspace") catch {case _:Throwable => 512}
    val noBiasStr = if (noBias) "True" else "False"
    Symbol.Convolution()()(Map("data" -> in, "kernel" -> kernel, "stride" -> stride, "pad" -> pad, "no_bias" -> noBiasStr, 
        "num_filter" -> numFilters, "workspace" -> workSpace.toString))
  }
  
  
  def getSymbol(sp: Config, inSymbol: Symbol, spType: String) : Symbol = {
    spType match {
        case "mx_conv" => mxConvolutionFromSpec(sp, inSymbol)
        case "pooling" => Symbol.Pooling()()(mapFromPoolSpec(sp, inSymbol))
        case "conv"    => convSymbolFromSpec(sp,inSymbol)
        case "convBN"  => convBNSymbolFromSpec(sp, inSymbol)
        case "inception" => inceptionSymbolFromSpec(sp, inSymbol)
        case "inceptionA" => inceptionASymbolFromSpec(sp, inSymbol)
        case "inceptionB" => inceptionBSymbolFromSpec(sp, inSymbol)
        case "fc" => fullyConnectedFromSpec(sp, inSymbol)
        case "flatten" => flattenFromSpec(sp, inSymbol)
        case "activation" => activationFromSpec(sp, inSymbol)
        case "dropout" => dropOutFromSpec(sp, inSymbol)
        case "softmax" => softMaxFromSpec(sp, inSymbol)
        case "batch_norm" => batchNormFromSpec(sp, inSymbol)
        case "resnetV2core" => resNetV2CoreFromSpec(sp, inSymbol)
        case a => throw new RuntimeException("Invalid network symbol type: " + a)
      }
  }
  /**
   * Simple 'interpreter' that maps configuration specification into MxNet symbol DAG
   */
  def symbolFromSpec(spec: Config, inName: String = "data", outName: String = "softmax") : Symbol = {  
    import scala.collection.JavaConversions._
    val data = Symbol.Variable("data")
    var lastName = ""
    try {
      val specList = spec.as[List[Config]]("mandolin.mx.specification")
    
    val finalMapping = 
      specList.foldLeft(Map[String,Symbol]("input" -> data)){case (curMap, sp) =>
      val spType = sp.as[String]("type")
      val spName = sp.as[String]("name")
      val inData = sp.as[String]("data") // input data
      val inSymbol = curMap(inData)
      val newSymbol = getSymbol(sp, inSymbol, spType)
      lastName = spName
      curMap + (spName -> newSymbol)
      }
      finalMapping(lastName)
    } catch {case e:Throwable =>
      // if the specification isn't a list, then assume it's in the "NEW" format
      val specObj = spec.as[Config]("mandolin.mx.specification")
      val layerNames = specObj.entrySet().toVector.map{x => x.getKey.split('.')(0)} 
      val nextMap = layerNames.toSet.foldLeft(Map():Map[String,String]){case (ac,v) =>        
        val inLayer = specObj.getConfig(v).getString("data")
        ac + (inLayer -> v)
        }      
      var building = true
      var prevName = "input"
      val buf = new collection.mutable.ArrayBuffer[(String,String)]
      while (building) {
        val current = nextMap.get(prevName)
        current match {case Some(c) => buf append ((prevName, c)); prevName = c case None => building = false}
      }
      val subSeqPairs = buf.toVector
      val lastName = subSeqPairs.last._2
      val finalMapping = subSeqPairs.foldLeft(Map[String,Symbol]("input" -> data)){case (curMap, sPair) =>
        val (prev,cur) = sPair
        val sp = specObj.getConfig(cur)
        val spType = sp.getString("type")
        val inSymbol = curMap(prev)
        val newSymbol = getSymbol(sp, inSymbol, spType)
        curMap + (cur -> newSymbol)
        }
      finalMapping(lastName)
    }
  }
}