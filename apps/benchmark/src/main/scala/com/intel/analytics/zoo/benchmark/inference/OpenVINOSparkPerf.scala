package com.intel.analytics.zoo.benchmark.inference

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import scopt.OptionParser
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.benchmark.inference.PerfUtils.{time, get_throughput}

import org.apache.spark.util.DoubleAccumulator


object OpenVINOSparkPerf {
  def main(argv: Array[String]): Unit = {
    val params = parser.parse(argv, new PerfParams).get

    val sc = NNContext.initNNContext("OpenVINO Perf on Spark")
    // Load model
    val model = new InferenceModel(1)
    val weight = params.model.substring(0,
      params.model.lastIndexOf(".")) + ".bin"
    model.doLoadOpenVINO(params.model, weight, params.batchSize)
    // BroadCast model
    val bcModel = sc.broadcast(model)
    // Register Accumulator
    val accPredict = new DoubleAccumulator
    sc.register(accPredict, "Predict Time")

    // Prepare warm up data
    val warm = (0 until 10).map { _ =>
      Tensor[Float](Array(1, params.batchSize, 224, 224, 3)).rand(0, 255)
    }

    // Prepare Test RDD
    val data = (0 until params.iteration).map { _ =>
      Tensor[Float](Array(1, params.batchSize, 224, 224, 3)).rand(0, 255)
    }
    val warmData = sc.parallelize(warm, numSlices = params.numInstance)
    val testData = sc.parallelize(data, numSlices = params.numInstance)
    // warm up
    testData.mapPartitions { p =>
      p.map { b =>
        // Get model
        val localModel = bcModel.value
        // Batch inference
        localModel.doPredict(b)
      }
    }

    // do the true performance
    accPredict.reset()
    val startTime = System.nanoTime()
    testData.mapPartitions { p =>
      p.map { b =>
        // Get model
        val localModel = bcModel.value
        // Batch inference
        val ps = System.nanoTime()
        localModel.doPredict(b)
        val predictTime = (System.nanoTime() - ps) / 1e6
        accPredict.add(predictTime)
        println(s"### Predict time $predictTime ms")
        (0 until 1)
      }
    }.count()
    val totalTime = (System.nanoTime() - startTime) / 1e6
    val averageBatch = accPredict.value / params.iteration
    val throughput = 1000 * params.batchSize * params.iteration / totalTime
    logger.info(s"Total Predict Time: $totalTime ms")
    logger.info(s"Total Throughput: $throughput FPS")
    logger.info(s"Average Batch Predict Time: $averageBatch ms")
  }


  val parser: OptionParser[PerfParams] = new OptionParser[PerfParams]("OpenVINO w/ Dnn Spark Model Performance Test") {
    opt[String]('m', "model")
      .text("serialized model, which is protobuf format")
      .action((v, p) => p.copy(model = v))
      .required()

    opt[Int]('n', "numInstance")
      .text("Number of instance")
      .action((v, p) => p.copy(numInstance = v))

    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))

    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
  }

  val logger = Logger.getLogger(getClass)
}
