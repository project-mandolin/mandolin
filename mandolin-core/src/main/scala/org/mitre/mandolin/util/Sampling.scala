package org.mitre.mandolin.util

/**
 * @author wellner
 */
object Sampling {
  
  
  
  /** Sample n indices, unique ranging from 0 to bigN-1, uniformly.
   *  This simple approach works well when n << bigN; O(n^2) technically */
  def sampleWithoutReplacement(bigN: Int, n: Int, initialResults: List[Int] = Nil) : List[Int] = {
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