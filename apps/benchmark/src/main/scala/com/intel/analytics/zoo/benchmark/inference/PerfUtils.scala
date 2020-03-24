package com.intel.analytics.zoo.benchmark.inference
import org.apache.log4j.Logger

object PerfUtils {

  val logger = Logger.getLogger(getClass)


  def timing[T](name: String)(f: => T): T = {
    val begin =System.currentTimeMillis()
    val result =f
    val end = System.currentTimeMillis()
    val cost = end - begin
    logger.info(s"$name time elapsed [${cost / 1000}] s, ${cost % 1000} ms")
    result
  }

  def time[R](block: => R, results: (Double) => String, iterations: Int, info: Boolean): Unit = {
    var i = 0
    while (i < iterations) {
      val start = System.nanoTime()
      block
      val end = System.nanoTime()

      val elapsed = (end - start) / 1e9
      if (info) {
        val throughput = results(elapsed)
        logger.info(s"Iteration $i , takes $elapsed s, throughput is $throughput imgs/sec")
      }

      i += 1
    }
  }

  def get_throughput(batchSize: Int)(elapsed: Double): String = {
    "%.2f".format(batchSize.toFloat / elapsed)
  }

}
