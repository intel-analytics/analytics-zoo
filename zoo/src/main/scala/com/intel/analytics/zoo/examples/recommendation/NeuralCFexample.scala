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

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import com.intel.analytics.zoo.optim.ZooOptimizer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val batchSize: Int = 80000,
                          val nEpochs: Int = 10,
                          val learningRate: Double = 1e-2,
                          val learningRateDecay: Double = 1e-6,
                          val nRdds: Int = 3
                         )

case class Rating(userId: Int, itemId: Int, label: Int)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratings, userCount, itemCount) = loadPublicData(sqlContext, param.inputDir)

    val isImplicit = false
    val ncf: NeuralCF[Float] = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 5,
      userEmbed = 20,
      itemEmbed = 20,
      hiddenLayers = Array(40, 20, 10))

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratings, userCount, itemCount)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2), seed = 1L)
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val beginTime = System.nanoTime()
    val optimMethod: Adam[Float] = new Adam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)

    //    val optimizer: ZooOptimizer[Float, MiniBatch[Float]] = ZooOptimizer(
    //      model = ncf,
    //      sampleRDD = trainRdds,
    //      criterion = ClassNLLCriterion[Float](),
    //      batchSize = param.batchSize,
    //      nEpochs = param.nEpochs,
    //      nRdds = param.nRdds)
    //
    //    optimizer
    //      .setOptimMethod(optimMethod)
    //      .optimize()
    //
    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = param.batchSize)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
      .optimize()

    val endTime = System.nanoTime()

    val results = ncf.predict(validationRdds)
    results.take(10).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(10).foreach(println)

    val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)

    userItemPairPrediction.take(10).foreach(println)

    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    val pairPredictionsDF = sqlContext.createDataFrame(userItemPairPrediction).toDF()
    val out = pairPredictionsDF.join(ratings, Array("userId", "itemId"))

    val correctCounts = out.filter(col("prediction") === col("label")).count()

    val accuracy = correctCounts.toDouble / out.count()
    println("accuracy: " + accuracy)
    println("training time: " + (endTime - beginTime) * (1e-9))
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userCount, itemCount)
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

}
