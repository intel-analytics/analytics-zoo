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

package com.intel.analytics.zoo.pipeline.estimator.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.python.api.EvaluatedResult
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.pipeline.estimator.{AbstractEstimator, Estimator}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._


object PythonEstimator {

  def ofFloat(): PythonEstimator[Float] = new PythonEstimator[Float]()

  def ofDouble(): PythonEstimator[Double] = new PythonEstimator[Double]()
}

class PythonEstimator[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def createEstimator(model: Module[T],
                      optimMethods: JMap[String, OptimMethod[T]],
                      modelDir: String = null): AbstractEstimator[T] = {
    if (modelDir == null) {
      Estimator(model, optimMethods.asScala.toMap)
    } else {
      Estimator(model, optimMethods.asScala.toMap, modelDir)
    }
  }

  def estimatorTrain(estimator: AbstractEstimator[T],
                     trainSet: FeatureSet[MiniBatch[T]],
                     criterion: Criterion[T] = null,
                     endTrigger: Trigger = null,
                     checkPointTrigger: Trigger = null,
                     validationSet: FeatureSet[MiniBatch[T]] = null,
                     validationMethod: JList[ValidationMethod[T]] = null): AbstractEstimator[T] = {
    val endTriggerOption = if (endTrigger == null) None else Some(endTrigger)
    val checkPointTriggerOption = if (checkPointTrigger == null) None else Some(checkPointTrigger)
    estimator.train(trainSet, criterion, endTriggerOption, checkPointTriggerOption, validationSet,
      if (validationMethod == null) null else validationMethod.asScala.toArray)
  }

  def estimatorEvaluate(estimator: AbstractEstimator[T],
                        validationSet: FeatureSet[MiniBatch[T]],
                        validationMethod: Array[ValidationMethod[T]]): JList[EvaluatedResult] = {
    val resultMap = estimator.evaluate(validationSet, validationMethod)
    processEvaluateResult(resultMap)
  }

  private def processEvaluateResult(resultMap: Map[ValidationMethod[T], ValidationResult])
  : JList[EvaluatedResult] = {
    resultMap.toList.map(result =>
      EvaluatedResult(result._2.result()._1, result._2.result()._2,
        result._1.toString())
    ).asJava
  }


}
