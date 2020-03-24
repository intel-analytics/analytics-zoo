
package com.intel.analytics.zoo.benchmark.inference

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory.Format
import com.intel.analytics.bigdl.nn.{Module, StaticGraph}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.benchmark.inference.PerfUtils.{time, get_throughput}
import org.apache.log4j.Logger
import scopt.OptionParser

object BigDLBLASPerf {
  def main(argv: Array[String]): Unit = {
    val params = parser.parse(argv, new PerfParams).get

    //val input = Tensor[Float](Array(params.batchSize, 3, 224, 224)).rand(-1, 1)

    val model = preprocess(Module.loadModule[Float](params.model), params.outputFormat)

    // Input tensor
    val tensorList = (0 until params.coreNumber).map { i =>
      Tensor[Float](Array(2, 3, 224, 224)).rand(-1, 1)
    }

    // model list
    val modelList = (0 until params.coreNumber).map{ i =>
      val newM = if (i == 0) {
        model
      } else {
        model.cloneModule()
      }
      // warm up
      time(newM.forward(tensorList(i)), get_throughput(2), 10, false)
      newM
    }

    // Perf
    logger.info("Begin parallel BLAS Perf")
    time((0 until params.coreNumber).indices.toParArray.foreach{ i =>
      // do the true performance
      time(modelList(i).forward(tensorList(i)), get_throughput(2), 10, false)
    }, get_throughput(2 * params.coreNumber * 10), params.iteration, info=true)
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

    opt[Int]('c', "core")
      .text("Core number")
      .action((v, p) => p.copy(coreNumber = v))
  }

  val logger = Logger.getLogger(getClass)
}
