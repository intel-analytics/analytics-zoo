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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.common.SeqToTensor
import org.apache.spark.ml.adapter.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame

import scala.reflect.ClassTag

/**
 * [[NNEvaluator]] evaluates the model performance based on prediction and label columns in a
 * DataFrame.
 */
class NNEvaluator[T: ClassTag] private[zoo] (
    val uid: String = Identifiable.randomUID("nnevaluator")
  )(implicit ev: TensorNumeric[T]) extends Params with HasPredictionCol with HasLabelCol {

  def setLabelCol(labelColName : String): this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def copy(extra: ParamMap): NNEvaluator[T] = {
    copyValues(
      new NNEvaluator(this.uid), extra)
  }

  /**
   * Evaluate with the validation methods
   */
  def evaluate(
      dataframe: DataFrame,
      vMethods: Array[ValidationMethod[T]]
    ): Array[(ValidationResult, ValidationMethod[T])] = {

    dataframe.select($(predictionCol), $(labelCol)).rdd.map { row =>
      val prediction = SeqToTensor().apply(Iterator(row.get(0))).next()
      val label = SeqToTensor().apply(Iterator(row.get(1))).next()
      vMethods.map { v =>
        v(prediction.reshape(Array(1) ++ prediction.size()),
          label.reshape(Array(1) ++ label.size()))
      }
    }.reduce((left, right) => {
      left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
  }
}


object NNEvaluator {
  /**
   * create a new NNEvaluator
   */
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): NNEvaluator[T] = {
    new NNEvaluator()
  }
}
