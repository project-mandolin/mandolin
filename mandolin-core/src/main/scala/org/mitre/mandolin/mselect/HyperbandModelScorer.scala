package org.mitre.mandolin.mselect

import akka.actor.{ActorRef, Actor}
import collection.mutable.PriorityQueue
import org.slf4j.LoggerFactory

class HyperbandModelScorer(modelConfigSpace: ModelSpace, acqFn: ScoringFunction, evalMaster: ActorRef,
                  sampleSize: Int, acqFnThreshold: Int, totalEvals: Int, concurrentEvals: Int, val maxIters: Int) 
extends ModelScorer(modelConfigSpace, acqFn, evalMaster, sampleSize, acqFnThreshold, totalEvals, concurrentEvals) { 

  import WorkPullingPattern._
  
  val eta = 3.0
  def logEta(x: Double) = math.log(x) / math.log(eta)
  
  private var targetRatiosTable : Map[(Int,Int),Double] = Map()  // map n_i,r_i => ratio
  private var currentCounts : Map[(Int,Int), Int] = Map()
  private var currentTotal = 0
  
  val sMax = logEta(maxIters.toDouble).toInt
  val B = (sMax+1) * maxIters
    
  // keeps eval results stratified by budget (i.e. number of iterations)
  // e.g. 0 = 1, 1 = 3, 2 = 9, 3 = 27, 4 = 81,...
  object ScoredModelConfigOrdering extends scala.math.Ordering[ScoredModelConfig] {
    def compare(a: ScoredModelConfig, b: ScoredModelConfig) = a.sc compare b.sc
  }
  
  val bandedResults = Array.tabulate(sMax){_ => new PriorityQueue[ScoredModelConfig]()(ScoredModelConfigOrdering) }
  
  var budgetToUseNext = 0
  
  def setUpRatioTable() = {
    var ratioCnt : Map[(Int,Int), Int] = Map()
    var total = 0
    for (s <- sMax to 0 by -1) {
      val n = math.ceil(B / maxIters * (math.pow(eta, s) / (s+1)))
      val r = maxIters * math.pow(eta, -s)
      for (i <- 0 to s) {
        val ni = (math.floor(n * math.pow(eta, -i))).toInt
        val ri = (r * math.pow(eta, i)).toInt
        val ri1 = if (i > 0) (r * math.pow(eta, i-1)).toInt else 0
        val curCnt = ratioCnt((ni,ri))
        total += ni // total number of evaluations
        ratioCnt += ((ri,ri1) -> (curCnt + ni)) // this updates        
      }
    }
    ratioCnt foreach {case (k,v) => currentCounts += (k -> 0); targetRatiosTable += (k -> (v.toDouble / total))}
  }
  
  def drawNextConfig() = {
    var maxDiff = -Double.MaxValue
    var config : Option[(Int,Int)] = None
    currentCounts foreach {case ((ri,x),v) =>
      val diff = targetRatiosTable((ri,x)) - v.toDouble / (currentTotal + 1.0)
      if (diff > maxDiff) {
        maxDiff = diff
        config = Some((ri,x))
      }
    }
    val prevEvalLevel = config.get._2
    val curEvalLevel  = config.get._1
    if (prevEvalLevel > 0) {
      // pick one of the non-extended configs, possibly using an acquisition function
      val index = logEta(prevEvalLevel).toInt
      val band = bandedResults(index)
      band.dequeue().mc.withBudgetAndSource(curEvalLevel,prevEvalLevel)
    } else {
      // in this case, draw using acquisition function (or randomly)
      val configs = getScoredConfigs(1, curEvalLevel)
      configs(0).withBudgetAndSource(curEvalLevel, prevEvalLevel)
    }
  }
  
  override def preStart() = {
    // send initial "random batch" of configs to evaluate
    val scored = getScoredConfigs(sampleSize) map (_._2)
    val epic = new Epic[ModelConfig] {
      override val iterator = scored.toIterator
    }
    evalMaster ! epic
    log.info("SCORER: Finished pre-start")
  }
  
  override def receive = {
    case ModelEvalResult(r) =>
      val scConf = r(0)
      val budgetUsed = scConf.mc.budget
      val prevBudget = scConf.mc.src    // "source" budget - i.e. number of resources spent to build model from which this one was drawn
      val ind = logEta(budgetUsed).toInt  // get the 'index' by taking the log here (e.g. budget = 81, eta=3 => index = 4)
      val curCnt = currentCounts((budgetUsed,prevBudget)) // count of # of evaluations with (budget, sourceBudget) 
      currentCounts += ((budgetUsed, prevBudget) -> (curCnt + 1))   // update count 
      currentTotal += 1
      bandedResults(ind) += scConf // add to priority queue
      // now need to select next to score and send for evaluation
      // keep an index to know which to select next...
      // how to handle mini-batches here .... ????
      val configs = Vector(drawNextConfig())
      val epic = new Epic[ModelConfig] {
          override val iterator = configs.toIterator
        }
      evalMaster ! epic
  }
  
  // just draw randomly
  def getScoredConfigs(size: Int, budget: Int) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom(budget)
    unscoredConfigs.toVector
  }
  
}