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

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.nn.DetectionOutputParam
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.python.api.{EvaluatedResult, JTensor, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.{Shape, Table}
import com.intel.analytics.zoo.common.{NNContext, PythonZoo}
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.models.anomalydetection.{AnomalyDetector, FeatureLabelIndex}
import com.intel.analytics.zoo.models.common.{KerasZooModel, Ranker, ZooModel}
import com.intel.analytics.zoo.models.image.common.{ImageConfigure, ImageModel}
import com.intel.analytics.zoo.models.image.objectdetection._
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelReader => IMCLabelReader}
import com.intel.analytics.zoo.models.image.objectdetection.common.ModuleUtil
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.{ByteRecord, Imdb}
import com.intel.analytics.zoo.models.image.objectdetection.common.evaluation.MeanAveragePrecision
import com.intel.analytics.zoo.models.image.objectdetection.common.loss.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.models.image.objectdetection.ssd.{RoiImageToSSDBatch, SSDMiniBatch, SSDVGG}
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, Recommender, UserItemFeature, UserItemPrediction}
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.models.seq2seq.{RNNDecoder, RNNEncoder, Seq2seq}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.models.textmatching.KNRM
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Embedding, WordEmbedding}
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet

import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame}


import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonZooModel {

  def ofFloat(): PythonZooModel[Float] = new PythonZooModel[Float]()

  def ofDouble(): PythonZooModel[Double] = new PythonZooModel[Double]()
}

class PythonZooModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

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

  def textClassifierCompile(
      model: TextClassifier[T],
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: JList[ValidationMethod[T]] = null): Unit = {
    model.compile(optimizer, loss,
      if (metrics == null) null else metrics.asScala.toList)
  }

  def textClassifierFit(
      model: TextClassifier[T],
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: TextSet): Unit = {
    model.fit(x, batchSize, nbEpoch, validationData)
  }

  def textClassifierPredict(
      model: TextClassifier[T],
      x: TextSet,
      batchPerThread: Int): TextSet = {
    model.predict(x, batchPerThread)
  }

  def textClassifierEvaluate(
      model: TextClassifier[T],
      x: TextSet,
      batchSize: Int): JList[EvaluatedResult] = {
    val resultArray = model.evaluate(x, batchSize)
    processEvaluateResult(resultArray)
  }

  private def processEvaluateResult(
    resultArray: Array[(ValidationResult, ValidationMethod[T])]): JList[EvaluatedResult] = {
    resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }.toList.asJava
  }

  def textClassifierSetCheckpoint(
      model: TextClassifier[T],
      path: String,
      overWrite: Boolean = true): Unit = {
    model.setCheckpoint(path, overWrite)
  }

  def textClassifierSetTensorBoard(
      model: TextClassifier[T],
      logDir: String,
      appName: String): Unit = {
    model.setTensorBoard(logDir, appName)
  }

  def createZooAnomalyDetector(
      featureShape: JList[Int],
      hiddenLayers: JList[Int],
      dropouts: JList[Double],
      model: AbstractModule[Activity, Activity, T]): AnomalyDetector[T] = {
    new AnomalyDetector[T](Shape(featureShape.asScala.toArray),
      hiddenLayers.asScala.toArray, dropouts.asScala.toArray)
      .addModel(model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]])
  }

  def loadAnomalyDetector(
      path: String,
      weightPath: String = null): AnomalyDetector[T] = {
      AnomalyDetector.loadModel(path, weightPath)
  }

  def standardScaleDF(df: DataFrame): DataFrame = {
    val fields = df.columns
    com.intel.analytics.zoo.models.anomalydetection.Utils.standardScale(df, fields)
  }

  def unroll(dataRdd: JavaRDD[JList[Double]],
             unrollLength: Int,
             predictStep: Int = 1): JavaRDD[JList[String]] = {
    val rdd: RDD[Array[Float]] = dataRdd.rdd.map(x => x.asScala.toArray.map(_.toFloat))
    val unrolled = AnomalyDetector.unroll[Float](rdd, unrollLength, predictStep)
    toUnrolledJavaRdd(unrolled)
  }

  private def toUnrolledJavaRdd(features: RDD[FeatureLabelIndex[Float]]): JavaRDD[JList[String]] = {
    features.map(x =>
      List(x.feature.map(x => x.mkString("|")).mkString(","), x.label.toString,
        x.index.toString).asJava).toJavaRDD()
  }

  private def toAnomaliesJavaRdd(anomaliesRdd: RDD[(Double, Double, Any)]): JavaRDD[JList[Any]] = {
    anomaliesRdd.map(x =>
      List(x._1, x._2, x._3.asInstanceOf[Any])
        .asJava).toJavaRDD()
  }

  def detectAnomalies(
      yTruth: JavaRDD[Object],
      yPredict: JavaRDD[Object],
      anomalySize: Int = 5): JavaRDD[JList[Any]] = {
    val out: RDD[(Double, Double, Any)] = AnomalyDetector.detectAnomalies[Double](
      yTruth.rdd.map(_.asInstanceOf[Double]), yPredict.rdd.map(_.asInstanceOf[Double]), anomalySize)
    toAnomaliesJavaRdd(out)
  }

  def zooModelSetEvaluateStatus(
    model: ZooModel[Activity, Activity, T]): ZooModel[Activity, Activity, T] = {
    model.setEvaluateStatus()
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
    Visualizer(labelMap.asScala.toMap, thresh, encoding, Visualizer.visualized) ->
      ImageBytesToMat(Visualizer.visualized) -> ImageMatToFloats(shareBuffer = false)
  }

  def getLabelMap(imageConfigure: ImageConfigure[T]): JMap[Int, String] = {
    if (imageConfigure.labelMap == null) null else imageConfigure.labelMap.asJava
  }

  def createImInfo(): ImInfo = {
    ImInfo()
  }

  def createDecodeOutput(): DecodeOutput = {
    DecodeOutput()
  }

  def createScaleDetection(): ScaleDetection = {
    ScaleDetection()
  }

  def createZooSSDVGG(classNum: Int, resolution: Int,
                      dataset: String, sizes: JList[Double],
                      postProcessParam: DetectionOutputParam): SSDVGG[T] = {
    if (sizes != null) {
      SSDVGG[T](classNum, resolution, dataset, sizes.asScala.toArray.map(_.toFloat), postProcessParam)
    } else {
      SSDVGG[T](classNum, resolution, dataset, null, postProcessParam)
    }
  }

  def createMeanAveragePrecision(use07metric: Boolean, normalized: Boolean = true,
                                 classes: JList[String]): MeanAveragePrecision = {
    new MeanAveragePrecision(use07metric, normalized, classes.asScala.toArray)
  }

  def createMultiBoxLossParam(locWeight: Double,
                              nClasses: Int,
                              shareLocation: Boolean,
                              overlapThreshold: Double,
                              bgLabelInd: Int,
                              useDifficultGt: Boolean,
                              negPosRatio: Double,
                              negOverlap: Double): MultiBoxLossParam = {
    MultiBoxLossParam(locWeight,
    nClasses,
    shareLocation,
    overlapThreshold,
    bgLabelInd,
    useDifficultGt,
    negPosRatio,
    negOverlap)
  }

  def createMultiBoxLoss(locWeight: Double,
                         nClasses: Int,
                         shareLocation: Boolean,
                         overlapThreshold: Double,
                         bgLabelInd: Int,
                         useDifficultGt: Boolean,
                         negPosRatio: Double,
                         negOverlap: Double): MultiBoxLoss[T] = {
    val param = MultiBoxLossParam(locWeight,
      nClasses,
      shareLocation,
      overlapThreshold,
      bgLabelInd,
      useDifficultGt,
      negPosRatio,
      negOverlap)
    MultiBoxLoss[T](param)
  }

  def toFloatTensor(jtensor: JTensor): Tensor[Float] = {
    Tensor[Float](jtensor.storage, jtensor.shape)
  }

  def createRoiImageToSSDBatch(batchSize: Int, convertLabel: Boolean = true,
                               partitionNum: Int = -1, keepImageFeature: Boolean = true,
                               inputKey: String = ImageFeature.floats): RoiImageToSSDBatch = {
    val pn = if (partitionNum == -1) None else Some(partitionNum)
    RoiImageToSSDBatch(batchSize, convertLabel, pn, keepImageFeature, inputKey)
  }

  def loadRoiSeqFilesToImageSet(url: String, sc: JavaSparkContext, partitionNum: Int= -1): ImageSet = {
    val pn = if (partitionNum <= 0) None else Some(partitionNum)
    Imdb.roiSeqFilesToImageSet(url, sc, pn)
  }

  def loadModelWeights(srcModel: Module[Float], targetModel: Module[Float],
                       matchAll: Boolean): Unit = {
    ModuleUtil.loadModelWeights(srcModel, targetModel, matchAll)
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
      mfEmbed: Int = 20,
      model: AbstractModule[Activity, Activity, T]): NeuralCF[T] = {
    new NeuralCF[T](userCount, itemCount, numClasses, userEmbed, itemEmbed,
      hiddenLayers.asScala.toArray, includeMF, mfEmbed)
      .addModel(model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]])
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
      wideBaseDims: JList[Int],
      wideCrossDims: JList[Int],
      indicatorDims: JList[Int],
      embedInDims: JList[Int],
      embedOutDims: JList[Int],
      continuousCols: JList[String],
      model: AbstractModule[Activity, Activity, T]): WideAndDeep[T] = {
    new WideAndDeep[T](modelType,
      numClasses,
      wideBaseDims.asScala.toArray,
      wideCrossDims.asScala.toArray,
      indicatorDims.asScala.toArray,
      embedInDims.asScala.toArray,
      embedOutDims.asScala.toArray,
      continuousCols.asScala.toArray,
      hiddenLayers.asScala.toArray)
        .addModel(model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]])
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

  def createZooKNRM(
      text1Length: Int,
      text2Length: Int,
      vocabSize: Int,
      embedSize: Int,
      embedWeights: JTensor = null,
      trainEmbed: Boolean = true,
      kernelNum: Int = 21,
      sigma: Double = 0.1,
      exactSigma: Double = 0.001,
      targetMode: String = "ranking",
      model: AbstractModule[Activity, Activity, T]): KNRM[T] = {
    KNRM[T](text1Length, text2Length, vocabSize, embedSize, toTensor(embedWeights),
      trainEmbed, kernelNum, sigma, exactSigma, targetMode, model)
  }

  def loadKNRM(
      path: String,
      weightPath: String = null): KNRM[T] = {
    KNRM.loadModel(path, weightPath)
  }

  def prepareEmbedding(
      embeddingFile: String,
      wordIndex: JMap[String, Int] = null,
      randomizeUnknown: Boolean = false,
      normalize: Boolean = false): JTensor = {
    val (_, _, embedWeights) = WordEmbedding.prepareEmbedding[T](
      embeddingFile, if (wordIndex!= null) wordIndex.asScala.toMap else null,
      randomizeUnknown, normalize)
    toJTensor(embedWeights)
  }

  def createZooSeq2seq(encoder: RNNEncoder[T],
    decoder: RNNDecoder[T],
    inputShape: JList[Int],
    outputShape: JList[Int],
    bridge: KerasLayer[Activity, Activity, T] = null,
    generator: KerasLayer[Activity, Activity, T] = null,
    model: AbstractModule[Table, Tensor[T], T]): Seq2seq[T] = {
    Seq2seq(encoder, decoder, toScalaShape(inputShape),
      toScalaShape(outputShape), bridge, generator, model)
  }

  def evaluateNDCG(
      ranker: Ranker[T],
      x: TextSet,
      k: Int,
      threshold: Double): Double = {
    ranker.evaluateNDCG(x, k, threshold)
  }

  def evaluateMAP(
      ranker: Ranker[T],
      x: TextSet,
      threshold: Double): Double = {
    ranker.evaluateMAP(x, threshold)
  }

  def seq2seqSetCheckpoint(model: Seq2seq[T],
    path: String,
    overWrite: Boolean = true): Unit = {
    model.setCheckpoint(path, overWrite)
  }

  def loadSeq2seq(path: String,
    weightPath: String = null): Seq2seq[T] = {
    Seq2seq.loadModel(path, weightPath)
  }

  def seq2seqCompile(
    model: Seq2seq[T],
    optimizer: OptimMethod[T],
    loss: Criterion[T],
    metrics: JList[ValidationMethod[T]] = null): Unit = {
    model.compile(optimizer, loss,
      if (metrics == null) null else metrics.asScala.toList)
  }

  def seq2seqFit(model: Seq2seq[T],
    x: JavaRDD[Sample],
    batchSize: Int,
    nbEpoch: Int,
    validationData: JavaRDD[Sample] = null): Unit = {
    model.fit(toJSample(x), batchSize, nbEpoch, toJSample(validationData))
  }

  def seq2seqInfer(model: Seq2seq[T],
    input: JTensor,
    startSign: JTensor,
    maxSeqLen: Int = 30,
    stopSign: JTensor = null,
    buildOutput: KerasLayer[Tensor[T], Tensor[T], T]): JTensor = {
    val result =
      model.infer(toTensor(input), toTensor(startSign), maxSeqLen,
        toTensor(stopSign), buildOutput)
    toJTensor(result)
  }

  def getModule(model: KerasZooModel[Activity, Activity, T]): KerasNet[T] = {
    model.model.asInstanceOf[KerasNet[T]]
  }
}
