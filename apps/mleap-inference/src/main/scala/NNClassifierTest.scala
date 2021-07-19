package com.intel.analytics.zoo.apps.mleap
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.NNClassifier
import ml.combust.bundle.BundleFile
import ml.combust.mleap.core.types._
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.combust.mleap.spark.SparkSupport._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.sql.SparkSession
import resource.managed

object NNClassifierTest {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("com").setLevel(Level.WARN)
    val conf = Engine.createSparkConf().setAppName("Test NNClassifier").setMaster("local[1]")
    val sc = NNContext.initNNContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._
    val irisDF = sc.textFile("/home/yuhao/workspace/github/hhbyyh/Test/MLeapTest/data/iris.data")
      .filter(_.nonEmpty)
      .map {l =>
        val splits = l.split(",")
        val f = splits.slice(0, 4).map(_.toFloat)
        val labelText = splits(4).trim
        val label = if (labelText == "Iris-setosa") {
          1.0
        } else if (labelText == "Iris-versicolor") {
          2.0
        } else if (labelText == "Iris-virginica") {
          3.0
        } else {
          4.0
        }
        (f, label)
      }.toDF("features", "label")

    val Array(trainingDF, validationDF) = irisDF.randomSplit(Array(0.8, 0.2))

    val model = new Sequential().add(Linear[Float](4, 3)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = NNClassifier(model, criterion)
      .setOptimMethod(new Adam[Float]())
      .setLearningRate(0.001)
      .setBatchSize(16)
      .setMaxEpoch(100)

    val nnModel = classifier.fit(trainingDF)
    validationDF.cache()
    val resultDF = nnModel.transform(validationDF)
    println(s"transforming ${validationDF.count()} records in validationDF:")
    println("Spark DataFrame transform benchmark (seconds):")
    Seq(1, 10, 100).foreach { end =>
      val st = System.nanoTime()
      (1 to end).map { i =>
        nnModel.transform(validationDF).collect()
      }
      println(s" $end time: " + (System.nanoTime() - st) / 1e9)
    }
    println()

    // serialize model to MLeap bundle
    val pipeline = SparkUtil.createPipelineModel(uid = "pipeline", Array(nnModel))
    val bundlePath = "jar:file:/tmp/NNClassifierModel.zip"
    val sbc = SparkBundleContext().withDataset(resultDF)
    for(bf <- managed(BundleFile(bundlePath))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }

    // load the model we saved in the previous section
    val bundle = (for(bundleFile <- managed(BundleFile(bundlePath))) yield {
      bundleFile.loadMleapBundle().get
    }).opt.get

    // create a simple LeapFrame to transform
    val schema = StructType(StructField("features", ListType(BasicType.Float)),
      StructField("label", ScalarType.Double)).get
    val data = validationDF.collect().map(r => Row(r.getSeq[Float](0), r.getDouble(1)))
    val frame = DefaultLeapFrame(schema, data)

    // transform the LeapFrame using our pipeline
    val mleapPipeline = bundle.root
    val leapFrameResult = mleapPipeline.transform(frame).get
    println("leap frame transform result:")
    println(leapFrameResult.dataset.take(4).mkString("\n"))
    println()

    println("MLeap Frame transform benchmark (seconds):")
    Seq(1, 10, 100).foreach { end =>
      val st = System.nanoTime()
      (1 to end).map { i =>
        mleapPipeline.transform(frame).get
      }
      println(s" $end time: " + (System.nanoTime() - st) / 1e9)
    }
  }

}
