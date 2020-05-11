package com.intel.analytics.zoo.feature

import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkEnv}

import scala.reflect.ClassTag



abstract class ExecutorFeatureSet[T] extends DistributedFeatureSet[T] {

  protected val sc: SparkContext

  private val cachedPerExecutorRdd = createCachedPerExecutorRdd()

  private def createCachedPerExecutorRdd() = {
    val bm = SparkEnv.get.blockManager
    val locs = bm.master.getPeers(bm.blockManagerId)
      .map(bmId => s"executor_${bmId.host}_${bmId.executorId}")
    sc.makeRDD(locs.map(loc => (loc, Seq(loc)))).cache()
  }

  protected def getRDD(): RDD[String] = {
    cachedPerExecutorRdd
  }

}

class DataStoreFeatureSet[T: ClassTag](dataStore: DataStore[T]) extends ExecutorFeatureSet[T] {

  private val cachedDataStoreRdd = {
    val dataStoreBroadcasted = sc.broadcast(dataStore)
    val coreNumer = EngineRef.getCoreNumber()
    val nodeNumber = EngineRef.getNodeNumber()

    super.getRDD().mapPartitionsWithIndex { case (idx, iter) =>

      val context = new DataStorePartitionContext {
        override val partitionIndex: Int = idx
        override val numOfCores: Int = coreNumer
        override val numOfPartitions: Int = nodeNumber
      }

      val dataStore = dataStoreBroadcasted.value
      dataStore.init(iter, context)
      Iterator.single(dataStore)
    }.cache()
  }

  override protected val sc: SparkContext = SparkContext.getOrCreate()

  /**
    * Get the 'origin' RDD of the dataset.
    *
    * @return
    */
  override def originRDD(): RDD[_] = {
    cachedDataStoreRdd
  }

  /**
    * Get a sequence of data
    *
    * @param train if the data is used in train. If yes, the data sequence is a looped endless
    *              sequence, or it has a limited length.
    * @return data sequence
    */
  override def data(train: Boolean): RDD[T] = {
    cachedDataStoreRdd.mapPartitions { iter =>
      val store = iter.next()
      store.makeIterators(train)
    }
  }

  /**
    * Change the order of the data sequence from the data set
    */
  override def shuffle(): Unit = {
    cachedDataStoreRdd.foreachPartition(iter => iter.next().shuffle())
  }

  /**
    * Total size of the data set
    *
    * @return
    */
  override def size(): Long = {
    cachedDataStoreRdd.mapPartitions(iter => Iterator.single(iter.next().size())).reduce(_ + _)
  }
}

object DataStoreFeatureSet {

  def apply[T: ClassTag](dataStore: DataStore[T]): DataStoreFeatureSet[T] = {
    new DataStoreFeatureSet(dataStore)
  }
}
