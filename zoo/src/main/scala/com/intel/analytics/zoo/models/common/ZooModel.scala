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

package com.intel.analytics.zoo.models.common

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{Container, Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, SparseAbstractModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.net.GraphNet
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * The base class for models in Analytics Zoo.
 *
 * @tparam A Input data type.
 * @tparam B Output data type.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
abstract class ZooModel[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends Container[A, B, T] with SparseAbstractModule[T] {

  /**
   * Override this method to define a model.
   */
  protected def buildModel(): AbstractModule[A, B, T]

  /**
   * The defined model, either from buildModel() or loaded from file.
   */
  def model: AbstractModule[A, B, T] = {
    if (modules.isEmpty) {
      throw new RuntimeException("No model found")
    }
    require(modules.length == 1,
      s"There should be exactly one model but found ${modules.length} models")
    modules(0).asInstanceOf[AbstractModule[A, B, T]]
  }

  override def sparseParameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    sparseParameters(modules(0), weights, gradWeights)

    (weights.toArray, gradWeights.toArray)
  }

  private def sparseParameters(m: AbstractModule[Activity, Activity, T],
    weights: ArrayBuffer[Tensor[T]], gradW: ArrayBuffer[Tensor[T]]): Unit = {
    if (m.isInstanceOf[Container[Activity, Activity, T]]) {
      m.asInstanceOf[Container[Activity, Activity, T]].modules
        .foreach(sparseParameters(_, weights, gradW))
    } else if (m.isInstanceOf[SparseAbstractModule[T]]) {
      val params = m.asInstanceOf[SparseAbstractModule[T]].sparseParameters()
      if (params != null) {
        if (params._1 != null) params._1.foreach(weights += _)
        if (params._2 != null) params._2.foreach(gradW += _)
      }
    }
  }

  def build(): this.type = {
    modules += buildModel()
    this
  }

  def addModel(model: AbstractModule[A, B, T]): this.type = {
    modules += model
    this
  }

  /**
   * Save the model to the specified path.
   *
   * @param path The path to save the model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path to save weights. Default is null.
   * @param overWrite Whether to overwrite the file if it already exists. Default is false.
   */
  def saveModel(path: String,
                weightPath: String = null,
                overWrite: Boolean = false): this.type = {
    this.saveModule(path, weightPath, overWrite)
  }

  /**
   * Print out the summary of the model.
   */
  def summary(): Unit = {
    if (this.model.isInstanceOf[KerasNet[T]]) {
      model.asInstanceOf[KerasNet[T]].summary()
    }
    else {
      println(model.toString())
    }
  }

  /**
   * Predict for classes. By default, label predictions start from 0.
   *
   * @param x Prediction data, RDD of Sample.
   * @param batchSize Number of samples per batch. Default is 32.
   * @param zeroBasedLabel Boolean. Whether result labels start from 0.
   *                       Default is true. If false, result labels start from 1.
   */
  def predictClasses(
      x: RDD[Sample[T]],
      batchSize: Int = -1,
      zeroBasedLabel: Boolean = true): RDD[Int] = {
    KerasUtils.toZeroBasedLabel(zeroBasedLabel, model.predictClass(x, batchSize))
  }

  /**
   * Set the model to be in evaluate status, i.e. remove the effect of Dropout, etc.
   */
  def setEvaluateStatus(): this.type = {
    model.evaluate()
    this
  }

  override def updateOutput(input: A): B = {
    output = model.updateOutput(input)
    output
  }

  override def updateGradInput(input: A, gradOutput: B): A = {
    gradInput = model.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: A, gradOutput: B): Unit = {
    model.accGradParameters(input, gradOutput)
  }
}

object ZooModel {
  Model
  Sequential
  GraphNet
  WordEmbedding
  /**
   * Load an existing model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](path: String,
                             weightPath: String = null)(implicit ev: TensorNumeric[T]):
      ZooModel[Activity, Activity, T] = {
    Module.loadModule[T](path, weightPath).asInstanceOf[ZooModel[Activity, Activity, T]]
  }
}
