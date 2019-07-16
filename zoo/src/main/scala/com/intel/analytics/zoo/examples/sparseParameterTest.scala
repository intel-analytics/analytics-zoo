package com.intel.analytics.zoo.examples

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.optim.{Trigger, sudoSparseSGD}
import com.intel.analytics.bigdl.tensor.{SparseTensor, Tensor, sudoLookupTableSparse}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalDistriOptimizer
import org.apache.spark.rdd.RDD

object sparseParameterTest {
  def main(args: Array[String]): Unit = {
    val output1 = 0.0f
    val output2 = 1.0f
    var plusOne = 0.0f
    val nodeNumber = args(0).toInt
    val coreNumber = args(1).toInt
    val batchSize = coreNumber

    val prepareData: Int => (MiniBatch[Float]) = index => {
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

//    System.setProperty("bigdl.ModelBroadcastFactory",
//      "com.intel.analytics.bigdl.models.utils.ZooModelBroadcastFactory")
    val sc = NNContext.initNNContext("sparse parameter test Example")

    val rdd = sc.parallelize(1 to (256 * nodeNumber), nodeNumber).map(prepareData)

    val dataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train: Boolean): RDD[MiniBatch[Float]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }

    plusOne = 0.0f

    val layer = new sudoLookupTableSparse[Float]()

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
    require(sparseW.head.almostEqual(oriW.add(1.0f), 1e-8))
    println("sparseW: " + sparseW)
    println("sparseG: " + sparseG)
  }
}
