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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.SparseAbstractModule
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * SparseLinear is the sparse version of module Linear. SparseLinear has two different from Linear:
  * firstly, SparseLinear's input Tensor is a SparseTensor. Secondly, SparseLinear doesn't backward
  * gradient to next layer in the backpropagation by default, as the gradInput of SparseLinear is
  * useless and very big in most cases.
  *
  * But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
  * part of the gradient to next layer.
  *
  * @param inputSize the size the each input sample
  * @param outputSize the size of the module output of each sample
  * @param backwardStart backwardStart index, counting from 1
  * @param backwardLength backward length
  * @param withBias if has bias
  */
class InternalSparseLinear[T: ClassTag](
  inputSize: Int,
  outputSize: Int,
  backwardStart: Int = -1,
  backwardLength: Int = -1,
  withBias: Boolean = true,
  initWeight: Tensor[T] = null,
  initBias: Tensor[T] = null)(implicit ev: TensorNumeric[T]) extends SparseLinear[T](
  inputSize, outputSize, backwardStart, backwardLength, withBias, null, null,
  initWeight, initBias) with SparseAbstractModule[T] {

  sWeight = Array(weight)

  override def accGradParameters(
                                  input: Tensor[T],
                                  gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 2,
      "SparseLinear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    if (scaleW != 0) {
      sparseGradWeight = Array(SparseTensorUtils.mmSparseTensor(ev.fromType[Double](scaleW), gradOutput.t, input))
    }

    if (withBias && scaleB != 0) {
      gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t, addBuffer)
    }
    // TODO: support regularize
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(bias), Array(gradBias))
  }

  override def toString() : String = {
    s"nn.InternalSparseLinear($inputSize -> $outputSize)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[SparseLinear[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SparseLinear[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        backwardStart == that.backwardStart &&
        backwardLength == that.backwardLength
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), backwardStart, backwardLength)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object InternalSparseLinear {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    backwardStart: Int = -1,
    backwardLength: Int = -1,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T]): InternalSparseLinear[T] = {
    new InternalSparseLinear[T](inputSize, outputSize, backwardStart, backwardLength,
      withBias, initWeight, initBias)
  }
}
