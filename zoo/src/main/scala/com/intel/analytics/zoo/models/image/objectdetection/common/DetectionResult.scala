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

package com.intel.analytics.zoo.models.image.objectdetection.common

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils

class MeanAveragePrecision(use07metric: Boolean, normalized: Boolean = true, classes: Array[String])
  extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    val nClass = classes.length
    val out = BboxUtil.decodeBatchOutput(output.toTensor, nClass)
    val gt = target.toTensor[Float]
    var i = 0
    val result = new Array[(Int, Array[(Float, Int, Int)])](nClass)
    val gtAreas = Tensor[Float](gt.size(1))
    BboxUtil.getAreas(gt, gtAreas, 4, normalized)
    while (i < nClass) {
      val cls = classes(i)
      if (cls != "__background__") {
        result(i) = EvalUtil.evaluateBatch(out, gt, gtAreas, i,
          ovThresh = 0.5, use07Metric = use07metric, normalized)
      } else {
        result(i) = (0, Array[(Float, Int, Int)]())
      }
      i += 1
    }
    new DetectionResult(result, use07metric, classes)
  }

  override protected def format(): String = "PascalMeanAveragePrecision"

  override def clone(): MeanAveragePrecision = SerializationUtils.clone(this)
}

/**
 *
 * @param results each element is the result for one class
 * (count, (score, tp, fp))
 */
class DetectionResult(private var results: Array[(Int, Array[(Float, Int, Int)])],
  use07metric: Boolean, classes: Array[String])
  extends ValidationResult {

  override def result(): (Float, Int) = {
    val output = PascalVocEvaluator.meanAveragePrecision(results, use07metric, classes)
    val meanAveragePrecision = output.map(_._2).sum / output.length
    (meanAveragePrecision, 1)
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[DetectionResult]
    this.results = results.zip(otherResult.results).map(x => {
      (x._1._1 + x._2._1, x._1._2 ++ x._2._2)
    })
    this
  }

  override protected def format(): String = {
    val output = PascalVocEvaluator.meanAveragePrecision(results, use07metric,
      classes: Array[String])
    val meanAveragePrecision = output.map(_._2).sum / output.length
    var info = ""
    info += "~~~~~~~~\n"
    info += "Results:\n"
    output.foreach(res => info += s"AP for ${ res._1 } = ${ "%.4f".format(res._2) }\n")
    info += s"Mean AP = ${ "%.4f".format(meanAveragePrecision) }\n"
    info += "~~~~~~~~\n"
    info
  }
}
