package org.mitre.mandolin.util

/**
 * @author wellner
 */
object Sampling {
  
  
  def sampleWithoutReplacement(ftable: Array[Int], n: Int, initialResults: List[Int]) : List[Int] = {
    val ll = ftable.length    
    var na = initialResults.length
    var results = initialResults
    while (na < n) {
      var ri = util.Random.nextInt(ll)
      val toAdd = ftable(ri)
      if (!results.contains(toAdd)) {
        na += 1
        results = toAdd :: results
      }
    }
    results
  }
  
  /** Sample n indices, unique ranging from 0 to bigN-1, uniformly.
   *  This simple approach works well when n << bigN; O(n^2) technically */
  def sampleWithoutReplacement(bigN: Int, n: Int, initialResults: List[Int]) : List[Int] = {
    if ((n.toDouble / bigN) > 0.1) sampleWithoutReplacementSmall(bigN, n, initialResults) else {
    var results: List[Int] = initialResults
    val rv = util.Random
    var m = 0
    while (m < n) {
      val id = util.Random.nextInt(bigN)
      if (!results.contains(id)) {
        m += 1
        results = id :: results
      }
    }
    results
    }
  }
  
  def shuffle(a: Array[Int]) : Unit = {
    val ll = a.length
    var mx = ll
    var i = 0; while (i < mx) {
      if (util.Random.nextBoolean) {
        val ni = util.Random.nextInt(ll - i)
        val tt = a(i)
        a(i) = a(ni)
        a(ni) = tt
        mx -= 1
      }
      i += 1
    }
  }
  
  def sampleWithoutReplacementSmall(bigN: Int, n: Int, initialResults: List[Int] = Nil) : List[Int] = {
    val els = (for (i <- 0 until bigN if !initialResults.contains(i)) yield i).toList
    val elsArray = els.toArray
    shuffle(elsArray)
    var results = initialResults
    for (i <- 0 until n) results = elsArray(i) :: results
    results
  }
}