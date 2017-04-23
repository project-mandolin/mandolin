package org.mitre.mandolin.mselect

import akka.actor.{ActorRef, Actor}
import collection.mutable.PriorityQueue
import org.slf4j.LoggerFactory
import breeze.linalg.DenseMatrix

class HyperbandModelScorer(modelConfigSpace: ModelSpace, acqFn: ScoringFunction, evalMaster: ActorRef,
                  sampleSize: Int, acqFnThreshold: Int, totalEvals: Int, concurrentEvals: Int, maxIters: Int=81) 
extends ModelScorer(modelConfigSpace, acqFn, evalMaster, sampleSize, acqFnThreshold, totalEvals, concurrentEvals) { 

  import WorkPullingPattern._
  
  lazy val eta = 3.0
  def logEta(x: Double) = math.log(x) / math.log(eta)
  
  private var targetRatiosTable : Map[(Int,Int),Double] = Map()  // map n_i,r_i => ratio
  private var currentCounts : Map[(Int,Int), Int] = Map()
  private var currentTotal = 0
  val targetRatioVec = setUpRatioTable()
  
  lazy val sMax = logEta(maxIters.toDouble).toInt
  lazy val B = (sMax+1) * maxIters
    
  // keeps eval results stratified by budget (i.e. number of iterations)
  // e.g. 0 = 1, 1 = 3, 2 = 9, 3 = 27, 4 = 81,...
  object ScoredModelConfigOrdering extends scala.math.Ordering[ScoredModelConfig] {
    def compare(a: ScoredModelConfig, b: ScoredModelConfig) = a.sc compare b.sc
  }
  
  private lazy val bandedResults = Array.tabulate(sMax+1){_ => new PriorityQueue[ScoredModelConfig]()(ScoredModelConfigOrdering) }  
  
  def setUpRatioTable() = {
    val power = 1.1
    var ratioCnt : Map[(Int,Int), Int] = Map()
    var total = 0
    for (s <- sMax to 0 by -1) {
      val n = math.ceil(B / maxIters * (math.pow(eta, s) / (s+1)))
      val r = maxIters * math.pow(eta, -s)
      for (i <- 0 to s) {
        val ni = (math.floor(n * math.pow(eta, -i))).toInt
        val ri = (r * math.pow(eta, i)).toInt
        val ri1 = if (i > 0) (r * math.pow(eta, i-1)).toInt else 0
        val curCnt = ratioCnt.get((ri,ri1)).getOrElse(0)
        total += ni
        ratioCnt += ((ri,ri1) -> (curCnt + ni)) // this updates        
      }
    }
    ratioCnt foreach {case (k,v) =>       
      currentCounts += (k -> 0); 
      targetRatiosTable += (k -> (v.toDouble / total.toDouble))
    }
    targetRatiosTable.toVector
  }
  
  def drawNextConfig(kMat: Option[DenseMatrix[Double]]) = {
    var config : Option[(Int,Int)] = None
    var maxDiff = -Double.MaxValue
    currentCounts foreach {case (k,v) =>
      val target = targetRatiosTable(k)
      val curRatio = v.toDouble / (currentTotal.toDouble + 1.0)
      val diff = target - curRatio
      if (diff > maxDiff) {
        config = Some(k)
        maxDiff = diff
      }
    }
    
    log.info(">>>> Drawing new config <<<<<  ==> " + config.get)
    val prevEvalLevel = config.get._2
    val curEvalLevel  = config.get._1
    if (prevEvalLevel > 0) {
      // pick one of the non-extended configs, possibly using an acquisition function
      val index = logEta(prevEvalLevel).toInt
      val band = bandedResults(index)
      if (band.length > 3) {
        // rescore band of scored configs with acquisition function - HERE:
        val band1 = band.toVector
        band.clear()
        // rescore using acquisition function
        val band2 = band1 map {conf =>
          // apply scoring function - but consider currently evaluating models using new "K" which will pin down variance 
          val acqFnScore = kMat match {case Some(k) => acqFn.scoreWithNewK(conf.mc, k) case None => acqFn.score(conf.mc)}
          ((conf.sc + acqFnScore), conf)} sortWith {(a,b) => a._1 > b._1}
        val toEval = band2.head._2
        band2.tail foreach {case (_,c) => band += c}
        val toEvalMc = toEval.mc
        val acqScore1 = acqFn.score(toEvalMc)
        val acqScore = kMat match {case Some(k) => acqFn.scoreWithNewK(toEvalMc, k) case None => acqScore1}
        log.info("Evaluating config with curEvalLevel = " + curEvalLevel + " from previous level " + prevEvalLevel + " that had score = " + toEval.sc + " with AcqFun score = " + acqScore + " without K = " + acqScore1)
        val cc = currentCounts(config.get)
        currentCounts += (config.get -> (cc + 1))        
        toEvalMc.withBudgetAndSource(curEvalLevel,prevEvalLevel)
      } else { // fall back to drawing uniformly if we don't have enough
        val (k,k1) = config.get
        val cc = currentCounts((k,0))
        currentCounts += (config.get -> (cc + 1))
        val configs = getScoredConfigs(sampleSize, curEvalLevel, kMat)
        configs(0)._2.withBudgetAndSource(curEvalLevel, 0)
      }
    } else {      
      val cc = currentCounts((curEvalLevel,0))
      currentCounts += ((curEvalLevel,0) -> (cc + 1))
      // in this case, draw using acquisition function (or randomly)
      val configs = getScoredConfigs(sampleSize, curEvalLevel, kMat)
      val config = configs(0)._2
      configs(0)._2.withBudgetAndSource(curEvalLevel, prevEvalLevel)
    }
  }
  
  def drawNConfigs(n: Int) = {
    var candSet = currentlyEvaluating
    var toEval : List[ModelConfig] = Nil
    for (i <- 1 to n) {
      val updatedK = acqFn.getUpdatedK(candSet.toVector)
      val c = drawNextConfig(updatedK)
      toEval = c :: toEval
      candSet += c
    }
    toEval.toVector
  }
  
  override def preStart() = {
    // send initial random batch of configs to evaluate
    val scored = getScoredConfigs(sampleSize,1, None) map {_._2}
    val epic = new Epic[ModelConfig] {
      override val iterator = scored.toIterator
    }
    evalMaster ! epic
    log.info("SCORER: Finished pre-start")
  }
  
  override def receive = {
    case CurrentlyEvaluating(c) => currentlyEvaluating += c
    case ModelEvalResult(r) =>
      currentlyEvaluating -= r.mc   // remove this from set of currently evaluating model configs
      evalResults += r // keep all eval results
      receivedSinceLastScore += 1      
      log.info("accuracy:" + r.sc + " " + r.mc + "\n")
      outWriter.print("accuracy:" + r.sc + " " + r.mc + "\n")
      outWriter.flush()
      if (totalReceived >= totalEvals) {
        outWriter.close()
        val hours = (System.currentTimeMillis() - startTime) / 1000.0 / 60 / 60
        log.info(s"Total time for $totalEvals configs was $hours hours")
        System.exit(0)
      }      
      val scConf = r
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
      
      if (receivedSinceLastScore >= acqFnThreshold) {
        log.info("Training acquisition function")
        receivedSinceLastScore = 0
        val t = System.nanoTime()
        acqFn.train(evalResults)
        log.info("Acquisition training completed in " + ((System.nanoTime() - t) / 1E9) + " seconds")
        log.info("Using updated K based on " + currentlyEvaluating.size + " currently evaluating models")
        val batchSize = math.max((concurrentEvals / 3), acqFnThreshold)
        val configs = drawNConfigs(batchSize)
        val epic = new Epic[ModelConfig] {
          override val iterator = configs.toIterator
        }
        evalMaster ! epic
      }      
  }
  
  // just draw according to acquisition function
  def getScoredConfigs(size: Int, budget: Int, kMat: Option[DenseMatrix[Double]]) = {
    val unscoredConfigs = for (i <- 1 to size) yield modelConfigSpace.drawRandom(budget)
    (unscoredConfigs map { s =>
      val sc1 = kMat match {case Some(k) => acqFn.scoreWithNewK(s, k) case None => acqFn.score(s)}
      val sc = if (sc1.isNaN() || (sc1 > 10.0) || (sc1 < -10.0)) acqFn.score(s) else sc1
      (sc, s) }).toVector.sortWith((a, b) => a._1 > b._1)
  }
  
}