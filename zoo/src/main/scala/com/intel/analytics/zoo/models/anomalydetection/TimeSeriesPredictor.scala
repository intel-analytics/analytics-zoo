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

package com.intel.analytics.zoo.models.anomalydetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.{KerasZooModel, ZooModel}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * The anomaly detector model for sequence data based on LSTM.
 *
 * @param featureShape The input shape of features.
 * @param hiddenLayers Units of hidden layers of LSTM.
 * @param dropouts     Fraction of the input units to drop out. Float between 0 and 1.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */

case class MFeatureLabelIndex[T: ClassTag](feature1: Array[Array[T]], feature2:Array[T], label: Array[T], index: Long) {
  override def toString(): String =
      "[ feature1: " + feature1.map(x => x.mkString("|")).mkString(",") + "\n"+
      " feature2:" + feature2.mkString("|") +"\n" +
      " label:" + label.mkString("|") + "\n" +
      " index :" + index +"]"
}

class TimeSeriesPredictor[T: ClassTag] (
                                    val featureShape: Shape,
                                    val hiddenLayers: Array[Int] = Array(128, 64, 16),
                                    val dropouts: Array[Double] = Array(0.2, 0.2, 0.2))
                                   (implicit ev: TensorNumeric[T])
  extends KerasZooModel[Tensor[T], Tensor[T], T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val model = Sequential()
    model.add(InputLayer(inputShape = featureShape))

    for (i <- 0 to hiddenLayers.length - 1) {
      model.add(Dense(hiddenLayers(i)))
        .add(Dropout(dropouts(i)))
    }

    model.add(Dense(hiddenLayers.last))
      .add(Dropout(dropouts.last))
      .add(Dense(outputDim = 1))

    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

}



object TimeSeriesPredictor {
 /**
  * The factory method to create an anomaly detector for single time series
  * @param featureShape The input shape of features.
  * @param hiddenLayers Units of hidden layers of LSTM.
  * @param dropouts     Fraction of the input units to drop out. Float between 0 and 1.
  * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
  */
  def apply[@specialized(Float, Double) T: ClassTag](
      featureShape: Shape,
      hiddenLayers: Array[Int] = Array(8, 32, 15),
      dropouts: Array[Double] = Array(0.2, 0.2, 0.2))
      (implicit ev: TensorNumeric[T]): TimeSeriesPredictor[T] = {
    require(hiddenLayers.size == dropouts.size,
      s"size of hiddenLayers and dropouts should be the same")
    new TimeSeriesPredictor[T](featureShape, hiddenLayers, dropouts).build()
  }

 /**
  * Load an existing AnomalyDetector model (with weights).
  * @param path       The path for the pre-defined model.
  *                   Local file system, HDFS and Amazon S3 are supported.
  *                   HDFS path should be like "hdfs://[host]:[port]/xxx".
  *                   Amazon S3 path should be like "s3a://bucket/xxx".
  * @param weightPath The path for pre-trained weights if any. Default is null.
  * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
  */
  def loadModel[T: ClassTag](
      path: String,
      weightPath: String = null)(implicit ev: TensorNumeric[T]): TimeSeriesPredictor[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[TimeSeriesPredictor[T]]
  }

  def unroll[T: ClassTag](dataRdd: RDD[Array[T]],
                          encoderLength: Int,
                          decoderLength: Int): RDD[MFeatureLabelIndex[T]] = {

    val n = dataRdd.count()
    val indexRdd: RDD[(Array[T], Long)] = dataRdd.zipWithIndex()

    val unrollLength = encoderLength + decoderLength -1
    val offset: Int = unrollLength

    // RDD[index of record, label]
    val labelRdd: RDD[(Long, Array[T])] = indexRdd.filter(x=> x._2 >= offset).map(x => (x._2 - offset, Array(x._1(0))))

    // RDD[index of record, feature]
    val featureRdd: RDD[(Long, Array[Array[T]])] = indexRdd
      .flatMap(x => {
        val pairs: Seq[(Long, List[(Array[T], Long)])] = if (x._2 < unrollLength) {
          (0L to x._2).map(index => (index, List(x)))
        } else {
          (x._2 - unrollLength + 1 to x._2).map(index => (index, List(x)))
        }
        pairs
      }).reduceByKey(_ ++ _)
      .filter(x => x._2.size == unrollLength && x._1 <= n - unrollLength - 1)
      .map(x => {

        val data: Array[Array[T]] = x._2.sortBy(y => y._2).map(x => x._1).toArray

        (x._1, data)
      })
      .sortBy(x => x._1)

    val featureLabelIndex = featureRdd.join(labelRdd)
      .sortBy(x => x._1)
      .map(x => {

        val feature1 = x._2._1.slice(0, encoderLength)
        val feature2 = x._2._1.slice(encoderLength-1, unrollLength).map(x=> x(0))
        val label = feature2.slice(1, decoderLength) ++ x._2._2

        MFeatureLabelIndex(feature1,feature2,label,x._1)

      })

    featureLabelIndex
  }

  def toSampleRdd[T: ClassTag](rdd: RDD[MFeatureLabelIndex[T]])
                              (implicit ev: TensorNumeric[T]): RDD[Sample[T]] = {
    rdd.map(x => {

      val shape1: Array[Int] = Array(x.feature1.length, x.feature1(0).length)
      val data1 = x.feature1.flatten
      val feature1 = Tensor[T](data1, shape1)

      val shape2: Array[Int] = Array(x.feature2.length, 1)
      val feature2 = Tensor[T](x.feature2, shape2)

      val label = Tensor[T](x.label, Array(x.label.length))

      Sample[T](Array(feature1, feature2), Array(label))

    })

  }

  def toEncoderRdd[T: ClassTag](rdd: RDD[MFeatureLabelIndex[T]])
                               (implicit ev: TensorNumeric[T]): RDD[Sample[T]] = {
    rdd.map(x => {

      val shape1: Array[Int] = Array(x.feature1.length, x.feature1(0).length)
      val data1 = x.feature1.flatten
      val feature1 = Tensor[T](data1, shape1)

      val label = Tensor[T](x.label, Array(x.label.length))

      Sample[T](Array(feature1), Array(label))

    })

  }

  def trainTestSplit(unrolled: RDD[MFeatureLabelIndex[Float]], testSize: Float)
  : (RDD[Sample[Float]], RDD[Sample[Float]],RDD[Sample[Float]]) = {

    val totalSize = unrolled.count()
    val testSizeInt = (totalSize * testSize).toInt

    trainTestSplit(unrolled, testSizeInt)
  }

  def trainTestSplit(unrolled: RDD[MFeatureLabelIndex[Float]], testSize: Int = 1000):
  (RDD[Sample[Float]], RDD[Sample[Float]],RDD[Sample[Float]]) = {

    val cutPoint = unrolled.count() - testSize

    val train =toSampleRdd(unrolled.filter(x => x.index < cutPoint))
    val test = toSampleRdd(unrolled.filter(x => x.index >= cutPoint))
    val encoderRdd = toEncoderRdd(unrolled)

    (train, test, encoderRdd)
  }
}



