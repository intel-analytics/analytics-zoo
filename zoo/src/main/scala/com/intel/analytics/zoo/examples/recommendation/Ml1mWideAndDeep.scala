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

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalDistriOptimizer
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD

case class User(userId: Int, gender: String, age: Int, occupation: Int)

case class Item(itemId: Int, title: String, genres: String)

object Ml1mWideAndDeep {

  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, userDF, itemDF, userCount, itemCount) =
      loadPublicData(sqlContext, params.inputDir)

    ratingsDF.groupBy("label").count().show()
    val bucketSize = 100
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("age-gender"),
      wideCrossDims = Array(bucketSize),
      indicatorCols = Array("genres", "gender"),
      indicatorDims = Array(19, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(userCount, itemCount),
      embedOutDims = Array(64, 64),
      continuousCols = Array("age"))

//    RandomGenerator.RNG.setSeed(10)
    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 5,
      columnInfo = localColumnInfo)

//    val t = wideAndDeep.sparseParameters()._1.head
//    import com.intel.analytics.bigdl.tensor.Tensor
//    val t2 = Tensor[Float](Array[Float](0.016113281f,-0.19726562f,-0.056884766f,0.19921875f,0.072753906f,0.20410156f,-0.040039062f,0.021850586f,-0.061523438f,-0.1875f,0.09033203f,-0.06640625f,0.024658203f,-0.20605469f,0.21386719f,0.12890625f,-0.05419922f,-0.12597656f,0.08642578f,-0.15332031f,0.018920898f,0.0859375f,-0.023925781f,-0.19042969f,-0.0859375f,-0.10498047f,0.04736328f,0.0044555664f,0.063964844f,0.21289062f,0.15917969f,-0.15234375f,0.040527344f,-0.021728516f,-0.14941406f,0.043701172f,0.13671875f,-0.17285156f,-0.14648438f,-0.09082031f,0.068847656f,0.095214844f,-0.13867188f,0.021850586f,-0.00680542f,0.14550781f,0.091796875f,0.034423828f,-0.008056641f,-0.13574219f,-0.15429688f,0.049804688f,0.099121094f,0.16601562f,0.13378906f,0.007232666f,0.096191406f,0.05419922f,0.07373047f,0.0021820068f,-0.1015625f,0.17578125f,-0.096191406f,-0.037353516f,0.14648438f,0.015258789f,0.0067749023f,-0.067871094f,0.140625f,-0.15917969f,0.055908203f,0.06982422f,0.018066406f,0.1875f,0.13964844f,0.048583984f,0.076660156f,0.14746094f,0.083496094f,-0.12158203f,-0.12109375f,0.17382812f,0.171875f,-0.2109375f,-0.15820312f,-0.11816406f,-0.16113281f,-0.15820312f,-0.14941406f,0.17578125f,-0.06689453f,0.17675781f,-0.21191406f,0.03491211f,-0.15820312f,-0.17773438f,0.09716797f,-0.16503906f,0.14160156f,0.11669922f,-0.20898438f,0.100097656f,-0.18652344f,-0.17773438f,-0.16699219f,-0.061279297f,-0.087890625f,0.11767578f,-0.15722656f,-0.15820312f,0.07080078f,0.016235352f,-0.07910156f,0.10986328f,-0.07421875f,-0.09765625f,-0.14550781f,0.028686523f,-0.1875f,-0.010009766f,0.104003906f,0.024414062f,0.09082031f,-0.025512695f,
//      -0.0036621094f,0.083496094f,0.15429688f,0.09375f,0.0032806396f,0.11035156f,0.032714844f,-0.19921875f,-0.10595703f,0.076660156f,0.06738281f,-0.009765625f,0.06542969f,-0.171875f,-0.1875f,0.049072266f,-0.20019531f,0.14550781f,0.064941406f,0.10058594f,-0.07421875f,-0.076660156f,0.17871094f,-0.18554688f,0.080566406f,-0.19921875f,0.044921875f,0.025390625f,0.20019531f,-0.14550781f,0.14550781f,-0.100097656f,0.061523438f,-0.11376953f,-0.10546875f,-0.20703125f,-0.068359375f,-0.14941406f,-0.107421875f,-0.20019531f,-0.091796875f,0.20703125f,0.18261719f,-0.060058594f,-0.083496094f,0.13964844f,0.029296875f,-0.037841797f,-0.056884766f,-0.10644531f,0.12695312f,-0.11328125f,-0.20019531f,0.11767578f,-0.12207031f,0.07373047f,0.09033203f,0.08886719f,0.05493164f,0.15136719f,-0.13378906f,0.009460449f,0.11376953f,-0.024658203f,0.16992188f,0.023071289f,0.072753906f,0.06542969f,-0.12011719f,0.12402344f,-0.026855469f,0.16894531f,0.030151367f,-0.08203125f,-0.16699219f,-0.15625f,0.012145996f,0.107910156f,-0.16992188f,0.0115356445f,-0.20214844f,0.122558594f,-0.050048828f,-0.030151367f,0.10058594f,0.14550781f,-0.15917969f,0.015991211f,-0.032714844f,-0.107421875f,-0.06689453f,-0.18945312f,-0.03930664f,-0.17578125f,-0.16699219f,-0.13964844f,0.037109375f,-0.026123047f,0.19433594f,0.13378906f,-0.10888672f,0.17675781f,0.041748047f,0.032714844f,-0.0050964355f,-0.08984375f,-0.18847656f,0.06542969f,0.19824219f,0.059326172f,-0.171875f,0.0015411377f,-0.024536133f,0.20019531f,0.10986328f,0.13183594f,-0.08691406f,0.18457031f,-0.034179688f,0.049072266f,0.14648438f,0.20703125f,0.016601562f,0.092285156f,
//      0.20703125f,-0.21191406f,0.1015625f,-0.091308594f,-0.123046875f,0.14648438f,-0.14941406f,-0.024414062f,0.095703125f,-0.140625f,0.09277344f,-0.16601562f,0.12792969f,0.12792969f,0.13476562f,0.063964844f,-0.16601562f,7.5149536E-4f,0.17382812f,-0.092285156f,0.11816406f,0.05883789f,0.011352539f,-0.056640625f,0.068847656f,0.016967773f,-0.018432617f,0.11279297f,0.10595703f,0.14648438f,-0.0016403198f,-0.16992188f,-0.052246094f,0.0045776367f,-0.19238281f,0.0625f,0.11328125f,-0.14941406f,-0.19726562f,-0.21484375f,-0.16992188f,0.20117188f,0.15917969f,0.13183594f,0.052490234f,-0.09716797f,-0.14453125f,-0.095214844f,0.20214844f,-0.16308594f,-0.091308594f,-0.008300781f,0.12988281f,0.18554688f,-0.04321289f,0.17871094f,-0.055419922f,-0.11035156f,-0.08691406f,-0.045166016f,-0.03149414f,0.20214844f,-0.05908203f,-0.09667969f,-0.20898438f,0.18847656f,-0.14648438f,0.115722656f,0.008605957f,0.16503906f,-0.11328125f,0.15917969f,0.15234375f,-0.15234375f,-0.057861328f,0.12451172f,-0.18164062f,0.17578125f,-0.18847656f,0.14550781f,0.0859375f,0.091308594f,-0.17089844f,-0.08300781f,-0.1328125f,-0.06298828f,-0.15332031f,-0.11328125f,0.091308594f,-0.17675781f,-0.15332031f,-0.13085938f,0.087402344f,0.049804688f,0.17871094f,0.18554688f,-0.111816406f,0.08203125f,0.041748047f,0.02709961f,0.13769531f,-0.1015625f,0.046142578f,0.011657715f,0.07519531f,0.13085938f,-0.10595703f,-0.088378906f,-0.18554688f,-0.13574219f,0.09082031f,-0.20703125f,0.16503906f,0.10449219f,0.18457031f,-0.19824219f,0.08935547f,-0.028564453f,0.15039062f,0.14257812f,-0.14355469f,-0.20898438f,-0.15332031f,0.0039978027f,
//      -0.19824219f,-0.0087890625f,0.21484375f,-0.20214844f,0.119628906f,-0.20996094f,0.17382812f,-0.072265625f,0.11279297f,-0.18847656f,0.12890625f,0.042236328f,-0.12890625f,0.16601562f,0.05029297f,-0.037841797f,0.19824219f,-0.19824219f,0.03540039f,-0.18066406f,0.11328125f,-0.1796875f,0.110839844f,0.1328125f,0.06982422f,0.12890625f,0.16796875f,-0.20800781f,-0.016845703f,0.15527344f,-0.012145996f,0.17675781f,0.059570312f,-0.04272461f,0.13769531f,-0.091796875f,0.09277344f,0.14746094f,-0.17089844f,0.05029297f,0.20410156f,-0.15429688f,0.072753906f,-0.11669922f,0.103515625f,0.14648438f,0.13183594f,0.1484375f,0.18554688f,-0.12060547f,-0.18945312f,0.13378906f,-0.17089844f,0.01550293f,0.056884766f,0.047607422f,0.012451172f,0.010253906f,-0.008239746f,0.060058594f,-0.18359375f,-0.21289062f,-0.078125f,-0.20410156f,-0.084472656f,0.08154297f,0.08496094f,-0.14160156f,-0.11767578f,0.083496094f,-0.037841797f,-0.092285156f,0.13378906f,-0.19140625f,-0.0057373047f,0.03540039f,-0.13964844f,-0.087890625f,-0.051757812f,-0.076171875f,-0.095214844f,0.20117188f,0.059326172f,-0.13085938f,0.13574219f,-0.06542969f,0.05834961f,-0.16992188f,-3.4332275E-4f,-0.042236328f,-0.04711914f,-0.17480469f,0.20410156f,0.0020751953f,0.08544922f,0.10449219f,-0.17871094f,-0.21289062f,-0.16308594f,-0.17675781f,-0.11376953f,-0.19140625f,-0.099609375f,0.16503906f,-0.064941406f,0.20898438f,-0.030639648f,0.008666992f,-0.01373291f,-0.12988281f,0.037597656f,0.18652344f,-0.03564453f,0.06640625f,0.045410156f,0.125f,-0.06298828f,0.0048217773f,0.17285156f,0.12402344f,-0.052734375f,-0.07080078f,0.071777344f,-0.16601562f,
//      -0.04711914f,0.17382812f,-0.122558594f,-0.21289062f,-0.18261719f,-0.18359375f,0.15234375f,-0.15820312f,-0.084472656f,0.19042969f,0.04321289f,-0.08496094f,-0.115722656f,-0.123535156f,-0.1796875f,-0.10498047f,0.18652344f,-0.17382812f,-0.028076172f,-0.042236328f,0.203125f,0.099609375f,0.21289062f,0.06347656f,0.118652344f,-0.107421875f,0.18847656f,-0.1171875f,-0.09082031f,-0.0018005371f,-0.095214844f,0.15722656f,-0.17382812f,0.057617188f,-0.063964844f,0.109375f,0.118652344f,-0.02709961f,0.037841797f,-0.04296875f,-0.0126953125f,0.095703125f,-0.1328125f,-0.05493164f,0.08935547f,0.022583008f,-0.15625f,-0.18066406f,0.17578125f,0.040527344f,-0.033203125f,-0.09423828f,-0.19824219f,0.10449219f,-0.13183594f,-0.08642578f,-0.04321289f,0.110839844f,0.05419922f,-0.11767578f,-0.028076172f,-0.21289062f,-0.17773438f,-0.15527344f,0.10058594f,-0.087890625f,-0.15527344f,-0.13085938f,0.05517578f,0.020996094f,-0.18945312f,-0.125f,0.20117188f,0.12011719f,-0.060058594f,-0.006591797f,-0.045654297f,0.060058594f,0.18457031f,-0.14257812f,-0.052490234f,0.18261719f,-0.12890625f,0.09326172f,-0.010498047f,-0.1484375f,-0.12988281f,0.096191406f,-0.009643555f,-0.13476562f,0.095214844f,0.14160156f,0.19335938f,-0.12988281f,0.088378906f,0.19824219f,0.100097656f,0.020507812f,0.08251953f,-0.119140625f,-0.2109375f,-0.123535156f,-0.17773438f,0.05517578f,0.010986328f,0.057373047f,-0.106933594f,-0.084472656f,-0.07128906f,0.13574219f,0.13476562f,-0.03491211f,-0.017211914f,0.13476562f,0.19726562f,-0.10107422f,0.053222656f,0.20800781f,-0.030517578f,0.16015625f,0.012817383f,-0.022094727f,-0.0703125f,-0.07080078f), Array(5, 124))
//    t.copy(t2)

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, userDF, itemDF, localColumnInfo, params.modelType)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)
    val trainBatch = DataSet.rdd(trainpairFeatureRdds.map(x => x.sample).cache()) ->
      SampleToMiniBatch[Float](params.batchSize)
    val validationBatch = DataSet.rdd(validationpairFeatureRdds.map(x => x.sample).cache()) ->
      SampleToMiniBatch[Float](params.batchSize)

//    val optimMethod = new Adam[Float](
//      learningRate = 1e-2,
//      learningRateDecay = 1e-5)

    val optimMethod = new Adagrad[Float](0.01)

    //    Due to https://github.com/intel-analytics/analytics-zoo/issues/1363,
//    have to use InternalDistriOptimizer to set optimMethod for sparse layer
//    wideAndDeep.compile(optimizer = optimMethod,
//      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
//      metrics = List(new Top1Accuracy[Float]())
//    )
//    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
//      nbEpoch = params.maxEpoch, validationData = validationRdds)

    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ZooModelBroadcastFactory")
    val optimizer = new InternalDistriOptimizer[Float](wideAndDeep,
      trainBatch.asInstanceOf[DistributedDataSet[MiniBatch[Float]]],
      SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false)
        .asInstanceOf[Criterion[Float]])
    optimizer.setSparseParameterProcessor(new SparseAdagrad[Float](0.01))
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.everyEpoch,
        validationBatch.asInstanceOf[DistributedDataSet[MiniBatch[Float]]],
        Array(new Top1Accuracy[Float], new Loss[Float]()))
      .setEndWhen(Trigger.maxEpoch(params.maxEpoch))

    optimizer.optimize()

    val results = wideAndDeep.predict(validationRdds)
    results.take(5).foreach(println)

    val resultsClass = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
    userItemPairPrediction.take(50).foreach(println)

    val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val userDF = sqlContext.read.text(dataPath + "/users.dat").as[String]
      .map(x => {
        val line = x.split("::")
        User(line(0).toInt, line(1), line(2).toInt, line(3).toInt)
      }).toDF()
    val itemDF = sqlContext.read.text(dataPath + "/movies.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Item(line(0).toInt, line(1), line(2).split('|')(0))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userDF, itemDF, userCount, itemCount)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      userDF: DataFrame,
                      itemDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {

    // age and gender as cross features, gender its self as wide base features
    val genderUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
    val bucketUDF = udf(Utils.buckBucket(100))
    val genresList = Array("Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi")
    val genresUDF = udf(Utils.categoricalFromVocabList(genresList))

    val userDfUse = userDF
      .withColumn("age-gender", bucketUDF(col("age"), col("gender")))
      .withColumn("gender", genderUDF(col("gender")))

    // genres as indicator
    val itemDfUse = itemDF
      .withColumn("genres", genresUDF(col("genres")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else ratingDF

    // userId, itemId as embedding features
    val joined = unioned
      .join(itemDfUse, Array("itemId"))
      .join(userDfUse, Array("userId"))
      .select(unioned("userId"), unioned("itemId"), col("label"), col("gender"), col("age"),
        col("occupation"), col("genres"), col("age-gender"))

    val rddOfSample = joined.rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
