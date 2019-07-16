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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.{MSECriterion, Sequential, SoftMax}
import com.intel.analytics.bigdl.optim.{DistriOptimizer, Trigger, sudoSparseSGD}
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalDistriOptimizer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object SparseGradientsSpec {
  private val output1 = 0.0f
  private val output2 = 1.0f
  private var plusOne = 0.0f
  private val nodeNumber = 4
  private val coreNumber = 4
  Engine.init(nodeNumber, coreNumber, onSpark = true)

  private val batchSize = coreNumber

  private val prepareData: Int => (MiniBatch[Float]) = index => {
    val input = Tensor[Float](batchSize)
    val target = Tensor[Float]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, output1 + plusOne)
        input.setValue(i + 1, 1.0f)
      } else {
        target.setValue(i + 1, output2 + plusOne)
        input.setValue(i + 1, 0.0f)
      }
      i += 1
    }
    MiniBatch(input, target)
  }
}

class SparseGradientsSpec extends FlatSpec with Matchers with BeforeAndAfter {
  import SparseGradientsSpec._
  private var sc: SparkContext = _

  private var dataSet: DistributedDataSet[MiniBatch[Float]] = _

  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * nodeNumber), nodeNumber).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Float]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }

    plusOne = 0.0f
    System.setProperty("bigdl.check.singleton", false.toString)
    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Sparse matrix" should "be aggregated correctly" in {
    val indices = Array[Int](0, 0, 1, 2)
    val indices2 = Array[Int](0, 1, 0, 3)
    val value = Array[Float](2.0f, 4.0f, 1.0f, 2.0f)

    val t2_indices= Array[Int](0, 0, 1, 2)
    val t2_indices2= Array[Int](0, 3, 1, 2)
    val t2_value = Array[Float](1.0f, 4.0f, 2.0f, 3.0f)

    val tensor1 = Tensor.sparse[Float](Array(indices, indices2), value, Array(3, 4))
    val tensor2 = Tensor.sparse[Float](Array(t2_indices, t2_indices2),
      t2_value, Array(3, 4))

    val tensor3 = SparseTensorUtils.addSparseTensor(tensor1, tensor2)
      .asInstanceOf[SparseTensor[Float]]
    require(tensor3._indices.head.array().deep == Array(0, 0, 0, 1, 1, 2, 2).deep)
    require(tensor3._indices.last.array().deep == Array(0, 1, 3, 0, 1, 2, 3).deep)
    require(tensor3.storage().array().deep ==
      Array(3.0f, 4.0f, 4.0f, 1.0f, 2.0f, 3.0f, 2.0f).deep)
  }

  "Dense matrix * Sparse matrix" should "generate correct result" in {
    val sparseM = Tensor.sparse(Tensor[Float](3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor[Float](2, 3).range(1, 12, 2)

    val res = SparseTensorUtils.mmSparseTensor[Float](2.0f, denseM, sparseM).asInstanceOf[SparseTensor[Float]]
    require(res._indices.head.array().deep == Array(0, 0, 1, 1).deep)
    require(res._indices.last.array().deep == Array(0, 1, 0, 1).deep)
    require(res.storage().array().deep ==
      Array(10.0f, 6.0f, 22.0f, 18.0f).deep)
  }

  "Dense matrix * Sparse matrix" should "generate correct result2" in {
    val sparseM = Tensor.sparse(Tensor[Float](3, 2).setValue(2, 2, 1).setValue(3, 1, 1)
      .setValue(3, 2, 1))
    val denseM = Tensor[Float](2, 3).range(1, 12, 2)

    val res = SparseTensorUtils.mmSparseTensor[Float](1.0f, denseM, sparseM).asInstanceOf[SparseTensor[Float]]
    require(res._indices.head.array().deep == Array(0, 0, 1, 1).deep)
    require(res._indices.last.array().deep == Array(0, 1, 0, 1).deep)
    require(res.storage().array().deep ==
      Array(5.0f, 8.0f, 11.0f, 20.0f).deep)
  }

  "Train with sudoSparseEmbedding and SGD" should "be trained with good result" in {
    LoggerFilter.redirectSparkInfoLogs()
    val layer = new sudoLookupTableSparse[Float]()

    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ZooModelBroadcastFactory")

//    val oriW = Tensor.sparse(Tensor[Float](6, 5).setValue(1, 3, 1.5f)
//      .setValue(2, 2, 3.0f).setValue(4, 5, 2.0f).setValue(6, 1, 1.0f))
    val oriW = Tensor[Float](Array(6, 5)).rand()
    layer.setSparseParameters(Array(oriW.clone()), null)
    val optimizer = new InternalDistriOptimizer[Float](layer.asInstanceOf[Module[Float]],
      dataSet, new MSECriterion[Float]().asInstanceOf[Criterion[Float]])
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxIteration(1))
    optimizer.setSparseParameterProcessor(new sudoSparseSGD[Float]())
    optimizer.optimize()
    val (sparseW, sparseG) = layer.sparseParameters()

//    require(sparseW.asInstanceOf[SparseTensor[Float]]._indices.head.array().deep
//      == Array(0, 1, 3, 5).deep)
//    require(sparseW.asInstanceOf[SparseTensor[Float]]._indices.last.array().deep
//      == Array(2, 1, 4, 0).deep)
//    require(sparseW.asInstanceOf[SparseTensor[Float]]._values.array().deep
//      == Array(2.5f, 4.0f, 3.0f, 2.0f).deep)

    require(sparseW.head.almostEqual(oriW.add(1.0f), 1e-8))
    require(sparseG.head.asInstanceOf[SparseTensor[Float]]._indices.head.array().deep
      == Array(0, 0, 1, 2, 3, 4, 4, 5).deep)
    require(sparseG.head.asInstanceOf[SparseTensor[Float]]._indices.last.array().deep
      == Array(1, 2, 1, 3, 4, 0, 3, 0).deep)
    require(sparseG.head.asInstanceOf[SparseTensor[Float]]._values.array().deep
      == Array(4.0f, 12.0f, 32.0f, 12.0f, 16.0f, 8.0f, 8.0f, 8.0f).deep)
  }
}
