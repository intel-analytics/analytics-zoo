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

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, Trigger}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.pipeline.estimator.Estimator


class TFOptimizer(modelPath: String,
                  optimMethod: OptimMethod[Float],
                  trainData: FeatureSet[MiniBatch[Float]],
                  validationData: FeatureSet[MiniBatch[Float]] = null,
                  sessionConfig: SessionConfig = null) {
  private val trainer: TFTrainingHelper = if (sessionConfig != null)
                                            TFTrainingHelper(modelPath, sessionConfig.toByteArray())
                                          else TFTrainingHelper(modelPath)
  private val estimator: Estimator[Float] = Estimator(trainer, optimMethod, modelPath)

  def optimize(endTrigger: Trigger = Trigger.maxEpoch(1),
               checkPointTrigger: Trigger = null): Array[Tensor[Float]] = {
    if (checkPointTrigger != null)
      estimator.train(trainData, new IdentityCriterion(), Some(endTrigger), Some(checkPointTrigger) )
    else
      estimator.train(trainData, new IdentityCriterion(), Some(endTrigger), None)
    trainer.parameters()._1
  }
}