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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.net.TFNetForInference
import com.intel.analytics.zoo.pipeline.nnframes.{NNEstimator, NNModel}
import org.slf4j.LoggerFactory
import org.tensorflow.DataType


class SparkEstimator private (model: TFTrainingHelper)
  extends NNEstimator[Float](model, new IdentityCriterion()) {

  super.setOptimMethod(new FakeOptimMethod[Float]())

  override protected def wrapBigDLModel(m: Module[Float]): NNModel[Float] = {
    val tfnet = trainingHelperToTFNet(m.asInstanceOf[TFTrainingHelper])
    super.wrapBigDLModel(tfnet)
  }

  override def setOptimMethod(value: OptimMethod[Float]): SparkEstimator.this.type = {

    throw new Exception("optimMethod is not support in SparkEstimator")
  }

  private def trainingHelperToTFNet(model: TFTrainingHelper): TFNetForInference = {

    val outputTypes = Array.tabulate(model.predictionOutputs.length)(
      _ => TFUtils.tfdatatype2enum(DataType.FLOAT))
    val variables = model.variables ++ model.extraVariables
    val variableTypes = model.variableTypes ++ model.extraVariableTypes
    val variableAssignPlaceholders =
      model.variableAssignPlaceholders ++ model.extraVariableAssignPlaceholders
    val assignVariableOps = Array(model.assignVariableOp, model.assignExtraVariableOP)
    val initWeights = model.parameters()._1 ++ model.getExtraParameter()
    val initOp = model.initOp
    new TFNetForInference(model.graphRunner.clone().asInstanceOf[GraphRunner],
                          model.inputs,
                          model.inputTypes,
                          model.predictionOutputs,
                          outputTypes,
                          variables: Array[String],
                          variableTypes: Array[Int],
                          variableAssignPlaceholders: Array[String],
                          assignVariableOps: Array[String],
                          initWeights: Array[Tensor[Float]],
                          initOp)
  }
}

object SparkEstimator {

  val logger = LoggerFactory.getLogger(getClass)

  def apply(exportPath: String): SparkEstimator = {
    val model = TFTrainingHelper(exportPath)
    new SparkEstimator(model)
  }
}
