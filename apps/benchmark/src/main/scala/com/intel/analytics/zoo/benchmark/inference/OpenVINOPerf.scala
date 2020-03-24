
package com.intel.analytics.zoo.benchmark.inference

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.benchmark.inference.PerfUtils.{time, get_throughput}
import org.apache.log4j.Logger
import scopt.OptionParser

object OpenVINOPerf {
  def main(argv: Array[String]): Unit = {
    val params = parser.parse(argv, new PerfParams).get
    val model = new InferenceModel(1)
    val weight = params.model.substring(0,
      params.model.lastIndexOf(".")) + ".bin"
    model.doLoadOpenVINO(params.model, weight, params.batchSize)

    val input = Tensor[Float](Array(1, params.batchSize, 224, 224, 3)).rand(0, 255)

    // warm up
    time(model.doPredict(input), get_throughput(params.batchSize), 10, false)

    // do the true performance
    time(model.doPredict(input), get_throughput(params.batchSize), params.iteration, true)
  }


  val parser: OptionParser[PerfParams] = new OptionParser[PerfParams]("OpenVINO w/ Dnn Local Model Performance Test") {
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
