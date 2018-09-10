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

package com.intel.analytics.zoo.models.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.python.api.{PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.common.{ImageConfigure, ImageModel}
import com.intel.analytics.zoo.models.image.objectdetection._
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelReader => IMCLabelReader}
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, Recommender, UserItemFeature, UserItemPrediction}
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.layers.Embedding
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonZooModel {

  def ofFloat(): PythonZooModel[Float] = new PythonZooModel[Float]()

  def ofDouble(): PythonZooModel[Double] = new PythonZooModel[Double]()
}

class PythonZooModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def saveZooModel(
      model: ZooModel[Activity, Activity, T],
      path: String,
      weightPath: String = null,
      overWrite: Boolean = false): ZooModel[Activity, Activity, T] = {
    model.saveModel(path, weightPath, overWrite)
  }

  def createZooTextClassifier(
      classNum: Int,
      embedding: Embedding[T],
      sequenceLength: Int = 500,
      encoder: String = "cnn",
      encoderOutputDim: Int = 256,
      model: AbstractModule[Activity, Activity, T]): TextClassifier[T] = {
    TextClassifier[T](classNum, embedding, sequenceLength, encoder, encoderOutputDim, model)
  }

  def loadTextClassifier(
      path: String,
      weightPath: String = null): TextClassifier[T] = {
    TextClassifier.loadModel(path, weightPath)
  }

  def loadObjectDetector(path: String, weightPath: String = null): ObjectDetector[T] = {
    ObjectDetector.loadModel(path, weightPath)
  }

  def loadImageClassifier(path: String, weightPath: String = null): ImageClassifier[T] = {
    ImageClassifier.loadModel(path, weightPath)
  }

  def readPascalLabelMap(): JMap[Int, String] = {
    LabelReader.readPascalLabelMap().asJava
  }

  def readCocoLabelMap(): JMap[Int, String] = {
    LabelReader.readCocoLabelMap().asJava
  }

  def readImagenetLabelMap(): JMap[Int, String] = {
    IMCLabelReader.readImagenetlLabelMap().asJava
  }

  def imageModelPredict(model: ImageModel[T],
    image: ImageSet,
    config: ImageConfigure[T] = null): ImageSet = {
    model.predictImageSet(image, config)
  }

  def getImageConfig(model: ImageModel[T]): ImageConfigure[T] = {
    model.getConfig
  }

  def createImageConfigure(
      preProcessor: Preprocessing[ImageFeature, ImageFeature],
      postProcessor: Preprocessing[ImageFeature, ImageFeature],
      batchPerPartition: Int,
      labelMap: JMap[Int, String],
      paddingParam: PaddingParam[T]): ImageConfigure[T] = {
    val map = if (labelMap == null) null else labelMap.asScala.toMap
    ImageConfigure(preProcessor, postProcessor, batchPerPartition, map, Option(paddingParam))
  }

  def createVisualizer(labelMap: JMap[Int, String], thresh: Float = 0.3f,
                       encoding: String): Preprocessing[ImageFeature, ImageFeature] = {
    Visualizer(labelMap.asScala.toMap,
      thresh,
      encoding,
      Visualizer.visualized) ->
      ImageBytesToMat(Visualizer.visualized) -> ImageMatToFloats(shareBuffer = false)
  }

  def getLabelMap(imageConfigure: ImageConfigure[T]): JMap[Int, String] = {
    if (imageConfigure.labelMap == null) null else imageConfigure.labelMap.asJava
  }

  def createImInfo(): ImInfo = {
    ImInfo()
  }

  def createDummyGT(): DummyGT = {
    DummyGT()
  }

  def createDecodeOutput(): DecodeOutput = {
    DecodeOutput()
  }

  def createScaleDetection(): ScaleDetection = {
    ScaleDetection()
  }

  def createPaddingParam(): PaddingParam[T] = {
    PaddingParam()
  }

  def createZooNeuralCF(
      userCount: Int,
      itemCount: Int,
      numClasses: Int,
      userEmbed: Int = 20,
      itemEmbed: Int = 20,
      hiddenLayers: JList[Int],
      includeMF: Boolean = true,
      mfEmbed: Int = 20): NeuralCF[T] = {
    NeuralCF[T](userCount, itemCount, numClasses, userEmbed, itemEmbed,
      hiddenLayers.asScala.toArray, includeMF, mfEmbed)
  }

  def loadNeuralCF(
      path: String,
      weightPath: String = null): NeuralCF[T] = {
    NeuralCF.loadModel(path, weightPath)
  }

  def createZooWideAndDeep(
      modelType: String = "wide_n_deep",
      numClasses: Int,
      hiddenLayers: JList[Int],
      wideBaseCols: JList[String],
      wideBaseDims: JList[Int],
      wideCrossCols: JList[String],
      wideCrossDims: JList[Int],
      indicatorCols: JList[String],
      indicatorDims: JList[Int],
      embedCols: JList[String],
      embedInDims: JList[Int],
      embedOutDims: JList[Int],
      continuousCols: JList[String],
      label: String = "label"): WideAndDeep[T] = {
    val columnFeatureInfo = ColumnFeatureInfo(
      wideBaseCols = wideBaseCols.asScala.toArray,
      wideBaseDims = wideBaseDims.asScala.toArray,
      wideCrossCols = wideCrossCols.asScala.toArray,
      wideCrossDims = wideCrossDims.asScala.toArray,
      indicatorCols = indicatorCols.asScala.toArray,
      indicatorDims = indicatorDims.asScala.toArray,
      embedCols = embedCols.asScala.toArray,
      embedInDims = embedInDims.asScala.toArray,
      embedOutDims = embedOutDims.asScala.toArray,
      continuousCols = continuousCols.asScala.toArray,
      label = label)
    WideAndDeep[T](modelType, numClasses, columnFeatureInfo, hiddenLayers.asScala.toArray)
  }

  def loadWideAndDeep(
      path: String,
      weightPath: String = null): WideAndDeep[T] = {
    WideAndDeep.loadModel(path, weightPath)
  }

  def toUserItemFeatureRdd(featureRdd: JavaRDD[Array[Object]]): RDD[UserItemFeature[T]] = {
    featureRdd.rdd.foreach(x =>
      require(x.length == 3, "UserItemFeature should consist of userId, itemId and sample"))
    featureRdd.rdd.map(x =>
      UserItemFeature(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int],
        toJSample(x(2).asInstanceOf[Sample])))
  }

  def toPredictionJavaRdd(predictionRdd: RDD[UserItemPrediction]): JavaRDD[JList[Double]] = {
    predictionRdd.map(x =>
      List(x.userId.toDouble, x.itemId.toDouble, x.prediction.toDouble, x.probability)
        .asJava).toJavaRDD()
  }

  def predictUserItemPair(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]]): JavaRDD[JList[Double]] = {
    val predictionRdd = model.predictUserItemPair(toUserItemFeatureRdd(featureRdd))
    toPredictionJavaRdd(predictionRdd)
  }

  def recommendForUser(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]],
      maxItems: Int): JavaRDD[JList[Double]] = {
    val predictionRdd = model.recommendForUser(toUserItemFeatureRdd(featureRdd), maxItems)
    toPredictionJavaRdd(predictionRdd)
  }

  def recommendForItem(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]],
      maxUsers: Int): JavaRDD[JList[Double]] = {
    val predictionRdd = model.recommendForItem(toUserItemFeatureRdd(featureRdd), maxUsers)
    toPredictionJavaRdd(predictionRdd)
  }

  def getNegativeSamples(indexed: DataFrame): DataFrame = {
    Utils.getNegativeSamples(indexed)
  }

  def zooModelSummary(model: ZooModel[Activity, Activity, T]): Unit = {
    model.summary()
  }

  def zooModelPredictClasses(
      module: ZooModel[Activity, Activity, T],
      x: JavaRDD[Sample],
      batchSize: Int = 32,
      zeroBasedLabel: Boolean = true): JavaRDD[Int] = {
    module.predictClasses(toJSample(x), batchSize, zeroBasedLabel).toJavaRDD()
  }

}
