
package com.intel.analytics.zoo.benchmark.inference

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.benchmark.inference.PerfUtils.{time, get_throughput}


import com.intel.analytics.zoo.pipeline.api.net.TorchNet

import org.apache.log4j.Logger
import scopt.OptionParser

object TorchNetPerf {
  def main(argv: Array[String]): Unit = {
    val params = parser.parse(argv, new PerfParams).get
    val net = TorchNet(params.model)


    val input = Tensor[Float](Array(params.batchSize, 3, 224, 224)).rand(-1, 1)

    // warm up
    time(net.forward(input), get_throughput(params.batchSize), 10, false)

    // do the true performance
    time(net.forward(input), get_throughput(params.batchSize), params.iteration, true)
  }


  val parser: OptionParser[PerfParams] = new OptionParser[PerfParams]("TorchNet w/ Dnn Local Model Performance Test") {
    opt[String]('m', "model")
      .text("serialized model, which is protobuf format")
      .action((v, p) => p.copy(model = v))
      .required()

    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))

    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))

  }

  val logger = Logger.getLogger(getClass)
}
