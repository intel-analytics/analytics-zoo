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

package com.intel.analytics.zoo.examples.objectdetection.finetune.fasterrcnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.nn.{Graph, Module, Sequential}
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.image.objectdetection.common.{IOUtils, MeanAveragePrecision, ModuleUtil, OBUtils}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.FrcnnMiniBatch
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.FrcnnCriterion
import com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn.{PostProcessParam, PreProcessParam, VggFRcnn}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import scopt.OptionParser

import scala.io.Source

object Option {

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    optim: String = "sgd",
    checkpoint: Option[String] = None,
    preTrainModel: String = "./",
    stateSnapshot: Option[String] = None,
    className: String = "",
    learningRate: Double = 0.0001,
    learningRateDecay: Double = 0.0005,
    step: Int = 50000,
    maxEpoch: Int = 50,
    jobName: String = "Analytics Zoo Fasterrcnn Fine Tune Example",
    summaryDir: Option[String] = None,
    checkIter: Int = 200,
    nPartition: Int = 1,
    batchSize: Int = 4
  )

  val trainParser = new OptionParser[TrainParams]("Analytics Zoo Fasterrcnn Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
    opt[String]("preTrainModel")
      .text("pretrain model location")
      .action((x, c) => c.copy(preTrainModel = x))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("step")
      .text("step to decay learning rate")
      .action((x, c) => c.copy(step = x))
    opt[Int]('e', "maxEpoch")
      .text("iteration numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("learning rate decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[String]("optim")
      .text("optim method")
      .action((x, c) => c.copy(optim = x))
    opt[String]("class")
      .text("class names")
      .action((x, c) => c.copy(className = x))
    opt[Int]("checkIter")
      .text("checkpoint iteration")
      .action((x, c) => c.copy(checkIter = x))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
    opt[Int]('b', "batchSize")
      .text("batch size, has to be same with total cores")
      .action((x, c) => c.copy(batchSize = x))
      .required()
  }
}


object Train {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = new SparkConf().setAppName(param.jobName)
      val sc = NNContext.initNNContext(conf)
      val classNames = Source.fromFile(param.className).getLines().toArray

      val postParam = PostProcessParam(0.3f, classNames.length, false, 100, 0.05)
      val preParamTrain = PreProcessParam(param.batchSize, Array(400, 500, 600, 700))
      val preParamVal = PreProcessParam(param.batchSize, nPartition = param.batchSize)

      val pretrain = Module.loadModule(param.preTrainModel)
      val model = VggFRcnn(classNames.length, postParam)
      ModuleUtil.loadModelWeights(pretrain, model, false)

      val trainSet = IOUtils.loadFasterrcnnTrainSet(param.trainFolder, sc, preParamTrain,
        param.batchSize, param.nPartition)

      val valSet = IOUtils.loadFasterrcnnValSet(param.valFolder, sc, preParamVal,
        param.batchSize, param.nPartition)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        param.optim match {
          case "sgd" =>
            val learningRateSchedule = SGD.Step(param.step, 0.1)
            new SGD[Float](
              learningRate = param.learningRate,
              momentum = 0.9,
              dampening = 0.0,
              learningRateSchedule = learningRateSchedule,
              weightDecay = 0.0005)
          case "adam" =>
            new Adam[Float](learningRate = param.learningRate,
              learningRateDecay = param.learningRateDecay)
        }
      }

      val meanAveragePrecision = new MeanAveragePrecision(use07metric = true, normalized = false,
        classes = classNames)
      optimize(model, trainSet, valSet, param, optimMethod,
        Trigger.maxEpoch(param.maxEpoch), new FrcnnCriterion(), meanAveragePrecision)

      model.saveModule("./pretrain.model")
    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[FrcnnMiniBatch],
    valSet: DataSet[FrcnnMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger,
    criterion: Criterion[Float],
    validationMethod: ValidationMethod[Float]): Module[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = criterion
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.severalIteration(param.checkIter))
    }

    optimizer.overWriteCheckpoint()

    if (param.summaryDir.isDefined) {
      val trainSummary = TrainSummary(param.summaryDir.get, param.jobName)
      val validationSummary = ValidationSummary(param.summaryDir.get, param.jobName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      optimizer.setTrainSummary(trainSummary)
      optimizer.setValidationSummary(validationSummary)
    }
    optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.severalIteration(param.checkIter),
        valSet.asInstanceOf[DataSet[MiniBatch[Float]]],
        Array(validationMethod))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
