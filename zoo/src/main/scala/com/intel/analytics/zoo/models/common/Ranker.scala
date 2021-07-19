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

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.common.Relation
import com.intel.analytics.zoo.feature.text.{DistributedTextSet, LocalTextSet, TextFeature, TextSet}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag
import scala.util.Random

/**
 * Trait for Ranking models (e.g., TextMatcher and Ranker) that
 * provides validation methods with different metrics.
 */
trait Ranker[T] {

  import Ranker.logger

  def model: AbstractModule[Activity, Activity, T]

  implicit val tag: ClassTag[T]
  implicit val ev: com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric[T]

  // TODO: May need to provide more types for x if this it to be used by Recommender
  protected def evaluate(
      x: TextSet,
      metrics: (Tensor[T], Tensor[T]) => Double): Double = {
    val result = x match {
      case distributed: DistributedTextSet =>
        val rdd = distributed.rdd
        val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model)
        rdd.mapPartitions(partition => {
          val localModel = modelBroad.value()
          localModel.evaluate()
          partition.map(feature => {
            val input = feature.getSample.feature()
            val output = localModel.forward(input).toTensor[T]
            val target = feature.getSample.label().toTensor[T]
            metrics(output, target)
          })
        }).mean()
      case local: LocalTextSet =>
        val res = local.array.map(feature => {
          val input = feature.getSample.feature()
          val output = model.evaluate().forward(input).toTensor[T]
          val target = feature.getSample.label().toTensor[T]
          metrics(output, target)
        })
        res.sum / res.length
    }
    result
  }

  /**
   * Evaluate using mean average precision on TextSet.
   *
   * @param x TextSet. Each TextFeature should contain Sample with batch features and labels.
   *          In other words, each Sample should be a batch of records having both positive
   *          and negative labels.
   * @param threshold Double. If label > threshold, then it will be considered as
   *                  a positive record. Default is 0.0.
   */
  def evaluateMAP(
      x: TextSet,
      threshold: Double = 0.0): Double = {
    val map = evaluate(x, Ranker.map[T](threshold))
    logger.info(s"map: $map")
    map
  }

  /**
   * Evaluate using normalized discounted cumulative gain on TextSet.
   *
   * @param x TextSet. Each TextFeature should contain Sample with batch features and labels.
   *          In other words, each Sample should be a batch of records having both positive
   *          and negative labels.
   * @param k Positive integer. Rank position.
   * @param threshold Double. If label > threshold, then it will be considered as
   *                  a positive record. Default is 0.0.
   */
  def evaluateNDCG(
      x: TextSet,
      k: Int,
      threshold: Double = 0.0): Double = {
    val ndcg = evaluate(x, Ranker.ndcg[T](k, threshold))
    logger.info(s"ndcg@$k: $ndcg")
    ndcg
  }

  def predictRelationLists(
      relations: RDD[Relation],
      corpus1: TextSet,
      corpus2: TextSet): DistributedTextSet = {
    val relationLists = TextSet.fromRelationLists(relations, corpus1, corpus2)
    val rdd = relationLists.rdd
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model.evaluate())
    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value()
      partition.map(feature => {
        val input = feature.getSample.feature()
        val output = localModel.forward(input).toTensor[T]
        feature(TextFeature.predict) = output
        feature
      })
    })
    TextSet.rdd(result)
  }
}

object Ranker {

  val logger: Logger = Logger.getLogger(getClass)

  def ndcg[@specialized(Float, Double) T: ClassTag](
      k: Int, threshold: Double = 0.0)
    (implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) => Double = {

    require(k > 0, s"k for NDCG should be a positive integer, but got $k")

    def validate(output: Tensor[T], target: Tensor[T])
      (implicit ev: TensorNumeric[T]): Double = {
      require(output.size().length == 2 && output.size()(1) == 1,
        s"output should be of shape (batch, 1), but got ${output.size()}")
      require(target.size().length == 2 && target.size()(1) == 1,
        s"target should be of shape (batch, 1), but got ${target.size()}")
      val yTrue = target.squeezeNewTensor().toArray().map(ev.toType[Double])
      val yPred = output.squeezeNewTensor().toArray().map(ev.toType[Double])
      val c = Random.shuffle(yTrue.zip(yPred).toList)
      val c_g = c.sortBy(_._1).reverse
      val c_p = c.sortBy(_._2).reverse
      var idcg = 0.0
      var dcg = 0.0
      for (((g, p), i) <- c_g.zipWithIndex) {
        if (i < k && g > threshold) {
          idcg += math.pow(2.0, g) / math.log(2.0 + i)
        }
      }
      for (((g, p), i) <- c_p.zipWithIndex) {
        if (i < k && g > threshold) {
          dcg += math.pow(2.0, g) / math.log(2.0 + i)
        }
      }
      if (idcg == 0.0) 0.0 else dcg / idcg
    }

    validate
  }

  def map[@specialized(Float, Double) T: ClassTag](
      threshold: Double = 0.0)
    (implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) => Double = {

    def validate(output: Tensor[T], target: Tensor[T])
      (implicit ev: TensorNumeric[T]): Double = {
      require(output.size().length == 2 && output.size()(1) == 1,
        s"output should be of shape (batch, 1), but got ${output.size()}")
      require(target.size().length == 2 && target.size()(1) == 1,
        s"target should be of shape (batch, 1), but got ${target.size()}")
      val yTrue = target.squeezeNewTensor().toArray().map(ev.toType[Double])
      val yPred = output.squeezeNewTensor().toArray().map(ev.toType[Double])
      val c = Random.shuffle(yTrue.zip(yPred).toList).sortBy(_._2).reverse
      var s = 0.0
      var ipos = 0
      for (((g, p), i) <- c.zipWithIndex) {
        if (g > threshold) {
          ipos += 1
          s += ipos / (i + 1.0)
        }
      }
      if (ipos == 0) 0.0
      else s / ipos
    }

    validate
  }
}
