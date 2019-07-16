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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.SparseAbstractModule

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer

object BigDLWrapperUtils {
  def selectCopy[T: ClassTag](src: Tensor[T], srcIndex: Int, dst: Tensor[T]): Tensor[T] = {
    Recurrent.selectCopy(src, srcIndex, dst)
  }

  def copy[T: ClassTag](src: ArrayBuffer[Tensor[T]], dst: Tensor[T]): Unit = {
    Recurrent.copy(src, dst)
  }

  def getAndClearWeightBias[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    if (parameters.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters.length)
      val isQuantized = parameters.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters(0).storage.array())
        (parameters.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters.length) {
        if (parameters(i) != null) {
          val wb = parameters(i)
          wb.getTensorType match {
            case QuantizedType =>
              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              weightsBias(i) = if (isCompacted) {
                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
              } else {
                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
              }
          }
          i += 1
        }
      }
      // clear parameters
      clearTensor(parameters)

      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
                                      (implicit ev: TensorNumeric[T]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        if (tensors(i).getTensorType == QuantizedType) {
          tensors(i).toQuantizedTensor.release()
        }

        tensors(i).set()
      }
      i += 1
    }
  }

  def putSparseGWeightBias[T: ClassTag](
    broadcastWeightBias: Array[Tensor[T]],
    localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val localWeightBias = localModel.asInstanceOf[SparseAbstractModule[T]].sparseParameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        clearAndSet(localWeightBias(i), broadcastWeightBias(i))
      }
      i += 1
    }

    def clearAndSet(old: Tensor[T], other: Tensor[T]): Unit = {
      if (old.getTensorType == QuantizedType && other.getTensorType == QuantizedType) {
        val quantOld = old.asInstanceOf[QuantizedTensor[T]]
        val quantOther = other.asInstanceOf[QuantizedTensor[T]]
        if (quantOld.getNativeStorage != quantOther.getNativeStorage) {
          quantOld.release()
        }
      }
      old.set(other)
    }
  }
}
