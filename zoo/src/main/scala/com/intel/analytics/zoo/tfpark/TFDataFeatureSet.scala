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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature._
import org.tensorflow.DataType
import com.intel.analytics.zoo.tfpark.TFTensorNumeric.NumericByteArray


class TFDataStore(private val graph: Array[Byte],
                  private val initIteratorOp: String,
                  private val outputNames: Array[String],
                  private val outputTypes: Array[DataType],
                  private val shardIndexOp: String) extends DataStore[MiniBatch[Float]] {

  private var shardIndexValue: Int = -1
  private var shardNum: Int = -1
  private var coreNum: Int = -1
  private var graphRunner: GraphRunner = null

  override def init(upstreamData: Iterator[Any],
                    context: DataStorePartitionContext): Unit = {
    shardIndexValue = context.partitionIndex
    shardNum = context.numOfPartitions
    coreNum = context.numOfCores
    val configBytes = SessionConfig(intraOpParallelismThreads = coreNum).toByteArray()

    graphRunner = GraphRunner(graph, null, null, null, null, configBytes)
  }

  override def makeIterators(train: Boolean): Iterator[MiniBatch[Float]] = {
    TFDataStore.makeIterators(
      graphRunner,
      train,
      initIteratorOp,
      shardIndexValue,
      shardIndexOp,
      outputTypes.toVector,
      outputNames.toVector
    )
  }

  override def shuffle(): Unit = {
    // doing nothing, tf dataset itself already shuffled
  }

  override def size(): Int = {
    // tf dataset's size is not known in advance
    -1
  }
}

object TFDataStore {
  def apply(graph: Array[Byte],
            initIteratorOp: String,
            outputNames: Array[String],
            outputTypes: Array[Int],
            shardIndex: String): TFDataStore = {
    val types = outputTypes.map(TFUtils.tfenum2datatype)
    new TFDataStore(graph, initIteratorOp, outputNames, types, shardIndex)
  }

  private[zoo] def generateOutputTensors(types: Vector[DataType]) = {
    val outputs = Array.tabulate[Tensor[_]](types.length) { i =>
      if (types(i) == DataType.STRING) {
        Tensor[Array[Byte]]()
      } else {
        Tensor[Float]()
      }
    }
    outputs
  }

  private[zoo] def makeIterators(graphRunner: GraphRunner,
                                 train: Boolean,
                                 initOp: String,
                                 idx: Int,
                                 shardIdx: String,
                                 types: Vector[DataType],
                                 names: Vector[String]): Iterator[TFMiniBatch] = {
    def intiIterator(): Unit = {
      graphRunner.runTargets(Vector(initOp),
        inputs = Vector(Tensor.scalar[Float](idx.toFloat)),
        inputTypes = Vector(DataType.INT64),
        inputNames = Vector(shardIdx))
    }
    if (train) {
      new Iterator[TFMiniBatch] {

        override def hasNext(): Boolean = {
          true
        }

        private def getNext() = {
          val outputs = TFDataStore.generateOutputTensors(types)
          val outputVec = outputs.toVector
          try {
            graphRunner.runOutputs(outputVec, names, types)
          } catch {
            case _: java.lang.IndexOutOfBoundsException =>
              intiIterator()
              graphRunner.runOutputs(outputVec, names, types)
            case _: java.lang.IllegalStateException =>
              intiIterator()
              graphRunner.runOutputs(outputVec, names, types)
            case e: Throwable => throw e
          }
          outputs
        }

        override def next(): TFMiniBatch = {
          TFMiniBatch(getNext())
        }
      }
    } else {
      intiIterator()
      new Iterator[TFMiniBatch] {

        private var buffer: Array[Tensor[_]] = null
        override def hasNext(): Boolean = {
          if (buffer != null) {
            true
          } else {
            val (success, result) = getNext()
            if (success) {
              buffer = result
            }
            success
          }
        }

        private def getNext() = {
          val outputs = TFDataStore.generateOutputTensors(types)
          val outputVec = outputs.toVector
          val success = try {
            graphRunner.runOutputs(outputVec, names, types)
            true
          } catch {
            case _: java.lang.IndexOutOfBoundsException => false
            case e: Throwable => throw e
          }
          (success, outputs)
        }

        override def next(): TFMiniBatch = {
          if (hasNext()) {
            val result = TFMiniBatch(buffer)
            buffer = null
            result
          } else {
            throw new NoSuchElementException("Next on an empty iterator")
          }
        }
      }
    }
  }
}

object TFDataFeatureSet {
  def apply(graph: Array[Byte],
            initIteratorOp: String,
            outputNames: Array[String],
            outputTypes: Array[Int],
            shardIndex: String): DistributedFeatureSet[MiniBatch[Float]] = {
    val dataStore = TFDataStore(graph, initIteratorOp, outputNames, outputTypes, shardIndex)
    new DataStoreFeatureSet[MiniBatch[Float]](dataStore)
  }

}
