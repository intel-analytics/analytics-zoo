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

package com.intel.analytics.zoo.examples.dlrm

import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer


object JoinFillNA {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val catCols = (14 until 40 toArray).map(i => "_c" + i)
    val intCols = (1 until 14 toArray).map(i => "_c" + i)

    val dataPath = "/var/backups/dlrm/terabyte/"
//    val dataPath = "/home/kai/Downloads/dac_sample/"
    val modelPath = dataPath + "models/"
    val parquetPath = dataPath + "parquet/"
    val sc = NNContext.initNNContext("DLRM Preprocess")
    println("Spark default parallelism: " + sc.defaultParallelism) // total cores
    val sqlContext = SQLContext.getOrCreate(sc)

    val files = new ArrayBuffer[String]()
    for( i <- 0 to 22) {
      files.append(parquetPath + "day_" + i + ".parquet")
    }
    val start = System.nanoTime
    var df = sqlContext.read.parquet(files.toList:_*)
    for( i <- 14 to 39) {
      val colName = catCols(i - 14)
      val model = sqlContext.read.parquet(modelPath + i + ".parquet").drop("model_count")
        .withColumnRenamed("data", colName)
      val broadcastModel = broadcast(model)
      df = df.join(broadcastModel, df(colName) === broadcastModel(colName), joinType="left")
        .drop(colName)
        .withColumnRenamed("id", colName)
    }

    df = df.na.fill(0, cols = intCols ++ catCols)
    df.explain()
    val rdd = df.rdd
    val count = rdd.count()
    val end = System.nanoTime
    println("Total time: " + (end - start) / 1e9d)
    println("Total count: " + count)
    println("Total partitions: " + rdd.getNumPartitions)
    rdd.take(5).foreach(println)
    sc.stop()
  }
}