/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.vnni.openvino

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.Logger
import scopt.OptionParser


case class ResNet50PerfParams(model: String = "",
                              weight: String = "",
                              batchSize: Int = 4,
                              numBatch: Int = 1,
                              iteration: Int = 10,
                              onSpark: Boolean = false)

object Perf {

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    val parser = new OptionParser[ResNet50PerfParams]("ResNet50 Int8 Performance Test") {
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((v, p) => p.copy(model = v))
        .required()
      opt[String]('w', "weight")
        .text("The path to the int8 ResNet50 model weight")
        .action((v, p) => p.copy(weight = v))
      opt[Int]('b', "batchSize")
        .text("Batch size of input data")
        .action((v, p) => p.copy(batchSize = v))
      opt[Int]('n', "numBatch")
        .text("Num of Batch of input data")
        .action((v, p) => p.copy(numBatch = v))
      opt[Int]('i', "iteration")
        .text("Iteration of perf test. The result will be average of each iteration time cost")
        .action((v, p) => p.copy(iteration = v))
      opt[Unit]("onSpark")
        .text("run with spark or not")
        .action((_, p) => p.copy(onSpark = true))
    }

    parser.parse(args, ResNet50PerfParams()).foreach { param =>
      if (!param.onSpark) {
        System.setProperty("bigdl.localMode", "true")
      }

      val batchSize = param.batchSize
      val numBatch = param.numBatch
      val iteration = param.iteration
      // Random input data
      val batchInput = Tensor(Array(numBatch, batchSize, 224, 224, 3)).rand()
      Engine.init

      val model = new InferenceModel(1)

      model.doLoadOpenVINO(param.model, param.weight, param.batchSize)

      // Warm up
      List.range(0, 15).foreach { _ =>
        model.doPredict(batchInput)
      }

      // Benchmark start
      val predictStart = System.nanoTime()
      var averageLatency = 0L
      List.range(0, iteration).foreach { _ =>
        val start = System.nanoTime()
        model.doPredict(batchInput)
        val latency = System.nanoTime() - start
        averageLatency += latency
        logger.info(s"Iteration latency is ${latency / 1e6} ms")
        logger.info(s"Latency per image is ${latency / batchSize / 1e6} ms")
        val throughPut = "%.2f".format(numBatch.toFloat * batchSize / (latency / 1e9))
        logger.info(s"Iteration throughput is ${throughPut} FPS")
        logger.info(s"*****************************************************")
      }
      val totalTimeUsed = System.nanoTime() - predictStart
      val totalThroughput = "%.2f".format(batchSize * iteration
        * numBatch.toFloat / (totalTimeUsed / 1e9))
      logger.info(s"Average latency of $iteration iteration is " +
        s"${averageLatency / iteration / 1e6} ms")
      logger.info(s"Average latency per image is " +
        s"${averageLatency / iteration / batchSize / 1e6} ms")
      logger.info(s"Average throughput of $iteration iteration is " +
        s"$totalThroughput FPS")
    }
  }
}
