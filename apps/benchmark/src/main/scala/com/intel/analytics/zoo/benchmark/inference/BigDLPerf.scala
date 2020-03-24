
package com.intel.analytics.zoo.benchmark.inference

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory.Format
import com.intel.analytics.bigdl.nn.{Module, StaticGraph}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.benchmark.inference.PerfUtils.{time, get_throughput}
import org.apache.log4j.Logger
import scopt.OptionParser

object BigDLPerf {
  def main(argv: Array[String]): Unit = {
    val params = parser.parse(argv, new PerfParams).get

    val input = Tensor[Float](Array(params.batchSize, 3, 224, 224)).rand(-1, 1)

    val model = try {
      val tmpModel = preprocess(Module.loadModule[Float](params.model), params.outputFormat).quantize()
      tmpModel.forward(input)
      // Check quantize error
      tmpModel
    } catch {
      case _: Exception => {
        logger.info(s"Quantize error. Switch to FP32")
        preprocess(Module.loadModule[Float](params.model), params.outputFormat)
      }
    }


    // warm up
    time(model.forward(input), get_throughput(params.batchSize), 10, false)

    // do the true performance
    time(model.forward(input), get_throughput(params.batchSize), params.iteration, true)
  }


  private def preprocess(model: Module[Float], outputFormat: String): Module[Float] = {
    // for MobileNet, the last layer may be a conv, which has 4-D output
    val format = outputFormat match {
      case "nc" => Format.nc
      case "nchw" => Format.nchw
      case _ => throw new UnsupportedOperationException(s"only support nc and nchw")
    }

    model.asInstanceOf[StaticGraph[Float]].setOutputFormats(Array(format))
    model
  }


  val parser: OptionParser[PerfParams] = new OptionParser[PerfParams]("BigDL w/ Dnn Local Model Performance Test") {
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

    opt[String]('o', "outputFormat")
      .text("output format of model")
      .action((v, p) => p.copy(outputFormat = v))
  }

  val logger = Logger.getLogger(getClass)
}
