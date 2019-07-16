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
package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, SparseAbstractModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
  *LookupTable for multi-values.
  * Also called embedding_lookup_sparse in TensorFlow.
  *
  * The input of LookupTableSparse should be a 2D SparseTensor or two 2D sparseTensors.
  * If the input is a SparseTensor, the values are positive integer ids,
  * values in each row of this SparseTensor will be turned into a dense vector.
  * If the input is two SparseTensors, the first tensor should be the integer ids, just
  * like the SparseTensor input. And the second tensor is the corresponding
  * weights of the integer ids.
  *

  */
class sudoLookupTableSparse[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[T], T] with SparseAbstractModule[T] with Initializable {
  sWeight = Array(Tensor[T](6, 5).rand())
  sparseGradWeight = Array(Tensor[T]())

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    sparseGradWeight = if (input.value() == 1) {
      Array(Tensor.sparse(Tensor[Float](6, 5).setValue(1, 3, 1.5f)
        .setValue(2, 2, 3.0f).setValue(4, 5, 2.0f).setValue(6, 1, 1.0f)).asInstanceOf[Tensor[T]])
    } else {
      Array(Tensor.sparse(Tensor[Float](6, 5).setValue(1, 2, 0.5f)
        .setValue(2, 2, 1.0f).setValue(3, 4, 1.5f).setValue(5, 1, 1.0f).setValue(5, 4, 1.0f))
        .asInstanceOf[Tensor[T]])
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(Tensor[T](10, 10)), Array(Tensor[T](10, 10)))
  }
}
