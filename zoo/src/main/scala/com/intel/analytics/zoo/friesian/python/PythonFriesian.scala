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

package com.intel.analytics.zoo.friesian.python

import java.net.{DatagramSocket, InetAddress}

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.friesian.feature.Utils
import java.util.{List => JList}

import com.github.mjakubowski84.parquet4s.ParquetWriter
import org.apache.parquet.hadoop.ParquetFileWriter
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number, spark_partition_id, udf, log => sqllog, rand}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.LongType

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.math.pow

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  val numericTypes: List[String] = List("long", "double", "integer")

  def fillNa(df: DataFrame, fillVal: Any = 0, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns
    } else {
      columns.asScala.toArray
    }

    val cols_idx = cols.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if(idx == -1) {
        throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
      }
      idx
    })

    Utils.fillNaIndex(df, fillVal, cols_idx)
  }

  def fillNaInt(df: DataFrame, fillVal: Int = 0, columns: JList[String] = null): DataFrame = {
    val schema = df.schema
    val allColumns = df.columns

    val cols_idx = if (columns == null) {
      schema.zipWithIndex.filter(pair => pair._1.dataType.typeName == "integer")
        .map(pair => pair._2)
    } else {
      val cols = columns.asScala.toList
      cols.map(col_n => {
        val idx = allColumns.indexOf(col_n)
        if (idx == -1) {
          throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
        }
        if (schema(idx).dataType.typeName != "integer") {
          throw new IllegalArgumentException(s"Only columns of IntegerType are supported, but " +
            s"the type of column ${col_n} is ${schema(idx).dataType.typeName}")
        }
        idx
      })
    }

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for (idx <- cols_idx) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillVal)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, schema)
  }

  def generateStringIdx(df: DataFrame, columns: JList[String], frequencyLimit: String = null)
  : JList[DataFrame] = {
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    if (frequencyLimit != null) {
      val freq_list = frequencyLimit.split(",")
      for (fl <- freq_list) {
        val frequency_pair = fl.split(":")
        if (frequency_pair.length == 1) {
          default_limit = Some(frequency_pair(0).toInt)
        } else if (frequency_pair.length == 2) {
          freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
        }
      }
    }
    val cols = columns.asScala.toList
    cols.map(col_n => {
      val df_col = df
        .select(col_n)
        .filter(s"${col_n} is not null")
        .groupBy(col_n)
        .count()
      val df_col_filtered = if (freq_map.contains(col_n)) {
        df_col.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col.filter(s"count >= ${default_limit.get}")
      } else {
        df_col
      }

      df_col_filtered.cache()
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(Utils.getPartitionSize)
        .collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)

      val windowSpec = Window.partitionBy("part_id").orderBy("count")
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
      val df_row_number = df_with_part_id.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int) => {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
    }).asJava
  }

  def compute(df: DataFrame): Unit = {
    df.rdd.count()
  }

  def log(df: DataFrame, columns: JList[String], clipping: Boolean = true): DataFrame = {
    var resultDF = df
    val zeroThreshold = (value: Int) => {
      if (value < 0) 0 else value
    }

    val zeroThresholdUDF = udf(zeroThreshold)
    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      if (clipping) {
        resultDF = resultDF.withColumn(colName, sqllog(zeroThresholdUDF(col(colName)) + 1))
      } else {
        resultDF = resultDF.withColumn(colName, sqllog(col(colName)))
      }
    }
    resultDF
  }

  def clipMin(df: DataFrame, columns: JList[String], min: Int): DataFrame = {
    var resultDF = df
    val clipFunc = (value: Int) => {
      if (value < min) min else value
    }
    val clipFuncUDF = udf(clipFunc)
    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      resultDF = resultDF.withColumn(colName, clipFuncUDF(col(colName)))
    }
    resultDF
  }

  def ordinalShufflePartition(df: DataFrame): DataFrame = {
    val shuffledDF = df.withColumn("ordinal", (rand() * pow(2, 52)).cast(LongType))
      .sortWithinPartitions(col("ordinal")).drop(col("ordinal"))
    shuffledDF
  }

  def shufflePartition(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    val rdd = df.rdd
    val schema = df.schema
    val shuffledRDD = rdd.mapPartitions(iter => {
      Random.shuffle(iter.toSeq).toIterator
    })
    spark.createDataFrame(shuffledRDD, schema)
  }

  def shuffleSubPartition(df: DataFrame, partSize: Int = 1000000): DataFrame = {
    val spark = df.sparkSession
    val rdd = df.rdd
    val schema = df.schema
    val shuffledRDD = rdd.mapPartitions(iter => {
      val resIters = new ArrayBuffer[Iterator[Row]]()
      var buffer = new ArrayBuffer[Row]()
      iter.foreach(row => {
        if (buffer.size == partSize) {
          resIters.append(Random.shuffle(buffer).toIterator)
          buffer = new ArrayBuffer[Row]()
          buffer.append(row)
        }
        else {
          buffer.append(row)
        }
      })
      if (buffer.nonEmpty){
        resIters.append(Random.shuffle(buffer).toIterator)
      }
      resIters.toList.reduce((iter1, iter2) => iter1 ++ iter2)
    })
    spark.createDataFrame(shuffledRDD, schema)
  }

  def dfWriteParquet(df: DataFrame, path: String, mode: String = "overwrite"): Unit = {
    df.write.mode(mode).parquet(path)
  }

  // TODO: can convert schema to case class.
  // https://gist.github.com/yoyama/ce83f688717719fc8ca145c3b3ff43fd
  case class Record(y: Int, xInt: List[Double], xCat: List[Int])

  def dfSaveParquet(df: DataFrame, path: String, mode: String = "overwrite", partSize: Int = 100000):
  JList[JList[Any]] = {
    // TODO: handle mode
    val rdd = df.rdd
    val res = rdd.mapPartitionsWithIndex((idx, iter) => {
      val meta = new ArrayBuffer[List[Any]]()
      val socket = new DatagramSocket()
      socket.connect(InetAddress.getByName("8.8.8.8"), 10002)
      val ip = socket.getLocalAddress.getHostAddress
      var buffer = new ArrayBuffer[Row]()
      var subIndex = 0
      val writeOptions = ParquetWriter.Options(
        writeMode = ParquetFileWriter.Mode.OVERWRITE
      )
      for (row <- iter) {
        if (buffer.size == partSize) {
          val records = Random.shuffle(buffer.map(
            row => Record(row.getInt(0), row.getList[Double](1).asScala.toList, row.getList[Int](2).asScala.toList)))
          val fileName = "partition_" + idx + "_" + subIndex + "_" + buffer.size + ".parquet"
          ParquetWriter.writeAndClose(path + fileName, records, options = writeOptions)
          meta.append(List(idx, subIndex, partSize, fileName, ip))
          subIndex += 1
          buffer = new ArrayBuffer[Row]()
          buffer.append(row)
        }
        else {
          buffer.append(row)
        }
      }
      if (buffer.nonEmpty){
        val records = Random.shuffle(buffer.map(
          row => Record(row.getInt(0), row.getList[Double](1).asScala.toList, row.getList[Int](2).asScala.toList)))
        val fileName = "partition_" + idx + "_" + subIndex + "_" + buffer.size + ".parquet"
        ParquetWriter.writeAndClose(path + fileName, records, options = writeOptions)
        meta.append(List(idx, subIndex, buffer.size, fileName, ip))
      }
      meta.map(_.asJava).toIterator
    }).collect().toList.asJava
    res
  }
}
