package com.intel.analytics.zoo.examples.nnframes.stucturedStreaming

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scopt.OptionParser


object ImageStructuredStreamingExample {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    parser.parse(args, ImageStreamingInferenceParams()).foreach { params =>
      val sc = NNContext.initNNContext("imageinfer")
      val spark = SparkSession
        .builder.config(sc.getConf)
        .getOrCreate()

      val lines = spark.readStream
        .format("text")
        .option("path", params.streamingPath)
        .load()

      val transformer = ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
        ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageSetToSample()
      val model = Module.loadModule[Float](params.model)
      val featureTransformersBC = sc.broadcast(transformer)
      val modelBroadCast = ModelBroadcast[Float]().broadcast(sc, model.evaluate())

      val readImageUDF = udf ( (path: String) => {
        val featureSteps = featureTransformersBC.value.cloneTransformer()
        val localModel = modelBroadCast.value()
        val localImageSet = ImageSet.read(path, imageCodec = 1).transform(transformer)
        val prediction = localModel.predictImage(localImageSet.toImageFrame())
          .toLocal().array.map(_.predict()).head.asInstanceOf[Tensor[Float]].toArray()
        prediction.zipWithIndex.maxBy(_._1)._2
      })

      val imageDF = lines.withColumn("output", readImageUDF(col("value")))
      val query = imageDF.writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", false)
        .start()

      query.awaitTermination()
    }

  }

  private case class ImageStreamingInferenceParams(streamingPath: String = "",
                                             model: String = "")

  private val parser = new OptionParser[ImageStreamingInferenceParams]("ImageStreamingInference") {
    head("Image Inference for Structured Streaming")
    opt[String]("model")
      .text("path to the model file")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[String]("streamingPath")
      .text("folder that used to store the streaming paths, local file system only, i.e. file:///path")
      .action((x, c) => c.copy(streamingPath = x))
      .required()
  }

}

