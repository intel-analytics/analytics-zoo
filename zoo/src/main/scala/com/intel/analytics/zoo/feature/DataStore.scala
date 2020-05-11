package com.intel.analytics.zoo.feature

import scala.reflect.ClassTag

trait DataStorePartitionContext {
  val numOfPartitions: Int
  val partitionIndex: Int
  val numOfCores: Int
}

abstract class DataStore[T: ClassTag] extends java.io.Serializable {

  def init(upstreamData: Iterator[Any], context: DataStorePartitionContext): Unit

  def makeIterators(train: Boolean): Iterator[T]

  def shuffle(): Unit

  def size(): Int

}
