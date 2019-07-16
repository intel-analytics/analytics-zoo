/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{SparseTensorUtils, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class sudoSparseSGD[@specialized(Float, Double) T: ClassTag](
  var learningRate: Double = 1e-3,
  var learningRateDecay: Double = 0.0,
  var weightDecay: Double = 0.0,
  var momentum: Double = 0.0,
  var dampening: Double = Double.MaxValue,
  var nesterov: Boolean = false,
  var learningRateSchedule: LearningRateSchedule = Default(),
  var learningRates: Tensor[T] = null,
  var weightDecays: Tensor[T] = null
)(implicit ev: TensorNumeric[T])
  extends OptimMethod[T] {

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]), x: Tensor[T])
  : (Tensor[T], Array[T]) = {
    val (fx, dfdx) = feval(x)
//    val updateW = SparseTensorUtils.addSparseTensorValueByConstant(x, ev.fromType(1))
    val updateW = x.add(ev.fromType(1))
    (updateW, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.weightDecay = config.get[Double]("weightDecay").getOrElse(this.weightDecay)
    this.momentum = config.get[Double]("momentum").getOrElse(this.momentum)
    this.dampening = config.get[Double]("dampening").getOrElse(this.dampening)
    this.nesterov = config.get[Boolean]("nesterov").getOrElse(this.nesterov)
    this.learningRateSchedule = config.get[LearningRateSchedule]("learningRateSchedule")
      .getOrElse(this.learningRateSchedule)
    this.learningRates = config.get[Tensor[T]]("learningRates").getOrElse(this.learningRates)
    this.weightDecays = config.get[Tensor[T]]("weightDecays").getOrElse(this.weightDecays)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("decayParameters")
    state.delete("dfdx")
    state.delete("deltaParameters")
  }

  override def getLearningRate(): Double = this.learningRateSchedule.currentRate
}