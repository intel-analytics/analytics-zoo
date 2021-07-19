package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{IntType, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.{MleapFrame2Sample, MleapSavePredict, NNContext}
import com.intel.analytics.zoo.examples.recommendation.NeuralCFexample.assemblyFeature
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import ml.combust.mleap.core.types.{ScalarType, StructField, StructType}
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions.{lit, max}
import org.apache.spark.sql.functions._
import scopt.OptionParser

import scala.io.Source

class NCFMleapFrame2Sample extends MleapFrame2Sample {
  override def toSample(frame: DefaultLeapFrame): Array[Sample[Float]] = {

    frame.select("userId", "itemId", "label")
      .get
      .dataset
      .map(row => {
        val uid = row.getAs[Double](0).toInt + 1
        val iid = row.getAs[Double](1).toInt + 1

        val label = row.getAs[Int](2)
        val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

        Sample(feature, Tensor[Float](T(label)))
      }).toArray
  }
}

object MleapNCFExample extends MleapSavePredict {

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

    val (pipeline, df, ratings, userCount, itemCount) = loadPublicData(sqlContext, param.inputDir)

    val isImplicit = false
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 5,
      userEmbed = 20,
      itemEmbed = 20,
      hiddenLayers = Array(40, 20, 10))

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratings, userCount, itemCount)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = param.batchSize)

    val optimMethod = new Adam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
      .optimize()

    println("train finished")

    val modelDir = System.getProperty("user.dir") + "/model"

    saveModels[Float](df, pipeline, ncf, modelDir)

    inference(param.inputDir, modelDir)
  }

  def inference(dataDir: String, modelDir: String) = {

    val data = Source
      .fromFile(dataDir + "/ratings.dat", "UTF-8")
      .getLines
      .map(x => {
        val line = x.split("::")
        (line(0), line(1), line(2).toInt)
      }).take(100)
      .map(x => Row(x._1, x._2, x._3))
      .toSeq

    val schema: StructType = StructType(StructField("strUserId", ScalarType.String),
      StructField("strItemId", ScalarType.String), StructField("label", ScalarType.String)).get

    val mleapFrame = DefaultLeapFrame(schema, data)

    val result = loadNpredict(mleapFrame, modelDir, new NCFMleapFrame2Sample)

    result.foreach(println)
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (PipelineModel, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val df = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::")
        (line(0), line(1), line(2).toInt)
      }).toDF("strUserId", "strItemId", "label").limit(100000)

    val userIndexer = new StringIndexer().setInputCol("strUserId").setOutputCol("userId")
    val itemIndexer = new StringIndexer().setInputCol("strItemId").setOutputCol("itemId")

    val pipelineEstimator: Pipeline = new Pipeline()
      .setStages(Array(userIndexer, itemIndexer))

    val plModel: PipelineModel = pipelineEstimator.fit(df)
    val indexedDF = plModel.transform(df).withColumn("userId", col("userId") + 1)
      .withColumn("itemId", col("itemId") + 1)

    indexedDF.show()

    val minMaxRow = indexedDF.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getDouble(0).toInt, minMaxRow.getDouble(1).toInt)

    println(userCount + "," + itemCount)

    (plModel, df, indexedDF, userCount, itemCount)

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
      val uid = row.getAs[Double](0).toInt
      val iid = row.getAs[Double](1).toInt

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }
}
