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
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch,
MiniBatch, DistributedDataSet}
import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.{NNContext}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.estimator.Estimator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}

import scopt.OptionParser

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import scala.reflect.ClassTag

case class Record(
                   age: Int,
                   workclass: String,
                   fnlwgt: Int,
                   education: String,
                   education_num: Int,
                   marital_status: String,
                   occupation: String,
                   relationship: String,
                   race: String,
                   gender: String,
                   capital_gain: Int,
                   capital_loss: Int,
                   hours_per_week: Int,
                   native_country: String,
                   income_bracket: String
                 )

object CensusWideAndDeep {

  val recordSchema = StructType(Array(
    StructField("age", IntegerType, false),
    StructField("workclass", StringType, false),
    StructField("fnlwgt", IntegerType, false),
    StructField("education", StringType, false),
    StructField("education_num", IntegerType, false),
    StructField("marital_status", StringType, false),
    StructField("occupation", StringType, false),
    StructField("relationship", StringType, false),
    StructField("race", StringType, false),
    StructField("gender", StringType, false),
    StructField("capital_gain", IntegerType, false),
    StructField("capital_loss", IntegerType, false),
    StructField("hours_per_week", IntegerType, false),
    StructField("native_country", StringType, false),
    StructField("income_bracket", StringType, false)
  ))

  case class RecordSample[T: ClassTag](sample: Sample[T])

  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val batchSize = params.batchSize
    val maxEpoch = params.maxEpoch
    val modelType = params.modelType

    val conf = new SparkConf().setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (trainDf, valDf) =
      loadCensusData(sqlContext, params.inputDir)

    println(trainDf.show(10))

    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("edu", "mari", "rela", "work", "occ", "age_bucket"),
      wideBaseDims = Array(16, 7, 6, 9, 1000, 11),
      wideCrossCols = Array("edu_occ", "age_edu_occ"),
      wideCrossDims = Array(1000, 1000),
      indicatorCols = Array("work", "edu", "mari", "rela"),
      indicatorDims = Array(9, 16, 7, 6),
      embedCols = Array("occ"),
      embedInDims = Array(1000),
      embedOutDims = Array(8),
      continuousCols = Array("age", "education_num", "capital_gain",
        "capital_loss", "hours_per_week"))

    RandomGenerator.RNG.setSeed(1)
    val wideAndDeep = WideAndDeep.sequential[Float](
      params.modelType,
      numClasses = 2,
      columnInfo = localColumnInfo,
      hiddenLayers = Array(100, 75, 50, 25))

    val isImplicit = false
    val trainpairFeatureRdds =
      assemblyFeature(isImplicit, trainDf, localColumnInfo, params.modelType)

    val sample1 = trainpairFeatureRdds.take(10)

    val validationpairFeatureRdds =
      assemblyFeature(isImplicit, valDf, localColumnInfo, params.modelType)

    val optimMethods = if (modelType == "wide_n_deep") {
      Map("deepPart" -> new Adagrad[Float](0.001),
        "widePart" -> new Ftrl[Float](math.min(5e-3, 1 / math.sqrt(3049))))
    } else if (modelType == "wide") {
      Map("widePart" -> new Ftrl[Float](math.min(5e-3, 1 / math.sqrt(3049))))
    } else if (modelType == "deep") {
      Map("deepPart" -> new Adagrad[Float](0.001),
      "sparse" -> new SparseAdagrad[Float](0.001))
    } else {
      throw new IllegalArgumentException(s"Unkown modelType ${modelType}")
    }

    val sample2batch = SampleToMiniBatch(batchSize)
    val trainRdds = FeatureSet.rdd(trainpairFeatureRdds.map(x => x.sample).cache()) ->
      sample2batch
    val validationRdds = FeatureSet.rdd(validationpairFeatureRdds.map(x => x.sample).cache()) ->
      sample2batch

//    val optimizer2 = Optimizer(
//      model = wideAndDeep,
//      dataset = trainRdds,
//      criterion = ClassNLLCriterion[Float]())
//    optimizer2
//      .setOptimMethod(optimMethod)
//      .setValidation(Trigger.everyEpoch, validationRdds,
//        Array(new Top1Accuracy[Float], new Loss[Float]()))
//      .setEndWhen(Trigger.maxEpoch(maxEpoch))

//    val optimizer = new ZooOptimizer[Float](wideAndDeep,
//      trainRdds.asInstanceOf[DistributedDataSet[MiniBatch[Float]]],
//      ClassNLLCriterion[Float]())
//    optimizer.setSparseParameterProcessor(new SparseAdagrad[Float](0.001))
//      .setOptimMethod(optimMethods)
//      .setValidation(Trigger.everyEpoch, validationRdds,
//        Array(new Top1Accuracy[Float], new Loss[Float]()))
//      .setEndWhen(Trigger.maxEpoch(maxEpoch))
//
//    optimizer.optimize()

    val estimator = if (params.logDir.isDefined) {
      val logdir = params.logDir.get
      val appName = "/census_wnd"
      Estimator[Float](wideAndDeep, optimMethods, modelDir = logdir + appName)
    } else {
      Estimator[Float](wideAndDeep, optimMethods)
    }

    val (checkpointTrigger, testTrigger, endTrigger) =
      (Trigger.everyEpoch, Trigger.everyEpoch, Trigger.maxEpoch(maxEpoch))

    estimator.train(trainRdds, ClassNLLCriterion[Float](),
      Some(endTrigger),
      Some(checkpointTrigger),
      validationRdds,
      Array(new Top1Accuracy[Float],
        new Loss[Float]()))
  }

  def loadCensusData(sqlContext: SQLContext, dataPath: String): (DataFrame, DataFrame) = {
    import sqlContext.implicits._
    val training = sqlContext.sparkContext
      .textFile(dataPath + "/adult.data")
      .map(_.split(",").map(_.trim))
      .filter(_.size == 15).map(array =>
      Record(
        array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
        array(5), array(6), array(7), array(8), array(9),
        array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
      )
    ).toDF()

    val validation = sqlContext.sparkContext
      .textFile(dataPath + "/adult.test")
      .map(_.dropRight(1))  // remove dot at the end of each line in adult.test
      .map(_.split(",").map(_.trim))
      .filter(_.size == 15).map(array =>
      Record(
        array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
        array(5), array(6), array(7), array(8), array(9),
        array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
      )
    ).toDF()

    (training, validation)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      dataDf: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[RecordSample[Float]] = {
    val educationVocab = Array("Bachelors", "HS-grad", "11th", "Masters", "9th",
      "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
      "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
      "Preschool", "12th") // 16
    val maritalStatusVocab = Array("Married-civ-spouse", "Divorced", "Married-spouse-absent",
      "Never-married", "Separated", "Married-AF-spouse", "Widowed")
    val relationshipVocab = Array("Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
      "Other-relative") // 6
    val workclassVocab = Array("Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
      "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked") // 9
    val genderVocab = Array("Female", "Male")

    val ages = Array(18f, 25, 30, 35, 40, 45, 50, 55, 60, 65)

    val educationVocabUdf = udf(Utils.categoricalFromVocabList(educationVocab))
    val maritalStatusVocabUdf = udf(Utils.categoricalFromVocabList(maritalStatusVocab))
    val relationshipVocabUdf = udf(Utils.categoricalFromVocabList(relationshipVocab))
    val workclassVocabUdf = udf(Utils.categoricalFromVocabList(workclassVocab))
    val genderVocabUdf = udf(Utils.categoricalFromVocabList(genderVocab))

    val bucket1Udf = udf(Utils.buckBuckets(1000)(_: String))
    val bucket2Udf = udf(Utils.buckBuckets(1000)(_: String, _: String))
    val bucket3Udf = udf(Utils.buckBuckets(1000)(_: String, _: String, _: String))

    val ageBucketUdf = udf(Utils.bucketizedColumn(ages))

    val incomeUdf = udf((income: String) => if (income == ">50K") 2 else 1)

    val data = dataDf
      .withColumn("age_bucket", ageBucketUdf(col("age")))
      .withColumn("edu_occ", bucket2Udf(col("education"), col("occupation")))
      .withColumn("age_edu_occ", bucket3Udf(col("age_bucket"), col("education"), col("occupation")))
      .withColumn("edu", educationVocabUdf(col("education")))
      .withColumn("mari", maritalStatusVocabUdf(col("marital_status")))
      .withColumn("rela", relationshipVocabUdf(col("relationship")))
      .withColumn("work", workclassVocabUdf(col("workclass")))
      .withColumn("occ", bucket1Udf(col("occupation")))
      .withColumn("label", incomeUdf(col("income_bracket")))

    val rddOfSample = data.rdd.map(r => {
      RecordSample(Utils.row2SampleSequential(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
