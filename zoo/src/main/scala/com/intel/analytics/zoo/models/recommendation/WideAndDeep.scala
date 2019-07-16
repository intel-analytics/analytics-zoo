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

package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, SparseAbstractModule}
import com.intel.analytics.bigdl.nn.{CAddTable, Concat, Linear, LogSoftMax
  , LookupTable, LookupTableSparse, ParallelTable, ReLU, Sequential, Zeros}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._

import scala.reflect.ClassTag

/**
 * The same data information shared by the WideAndDeep model and its feature generation part.
 *
 * @param wideBaseCols   Data of wideBaseCols together with wideCrossCols will be fed into the
 *                       wide model.
 * @param wideBaseDims   Dimensions of wideBaseCols. The dimensions of the data in wideBaseCols
 *                       should be within the range of wideBaseDims.
 * @param wideCrossCols  Data of wideCrossCols will be fed into the wide model.
 * @param wideCrossDims  Dimensions of wideCrossCols. The dimensions of the data in wideCrossCols
 *                       should be within the range of wideCrossDims.
 * @param indicatorCols  Data of indicatorCols will be fed into the deep model as multi-hot vectors.
 * @param indicatorDims  Dimensions of indicatorCols. The dimensions of the data in indicatorCols
 *                       should be within the range of indicatorDims.
 * @param embedCols      Data of embedCols will be fed into the deep model as embeddings.
 * @param embedInDims    Input dimension of the data in embedCols. The dimensions of the data
 *                       in embedCols should be within the range of embedInDims.
 * @param embedOutDims   The dimensions of embeddings.
 * @param continuousCols Data of continuousCols will be treated as continuous values
 *                       for the deep model.
 * @param label          The name of the 'label' column. Default is 'label'.
 */
case class ColumnFeatureInfo(wideBaseCols: Array[String] = Array[String](),
                             wideBaseDims: Array[Int] = Array[Int](),
                             wideCrossCols: Array[String] = Array[String](),
                             wideCrossDims: Array[Int] = Array[Int](),
                             indicatorCols: Array[String] = Array[String](),
                             indicatorDims: Array[Int] = Array[Int](),
                             embedCols: Array[String] = Array[String](),
                             embedInDims: Array[Int] = Array[Int](),
                             embedOutDims: Array[Int] = Array[Int](),
                             continuousCols: Array[String] = Array[String](),
                             label: String = "label") extends Serializable{
  override def toString: String = {
    "wideBaseCols:" + wideBaseCols.mkString(",") +"\n" +
    "wideBaseDims:" + wideBaseDims.mkString(",") + "\n" +
    "wideCrossCols:" + wideCrossCols.mkString(",") +"\n" +
    "wideCrossDims:" + wideCrossDims.mkString(",") +"\n" +
    "indicatorCols:" + indicatorCols.mkString(",") + "\n" +
    "indicatorDims:" + indicatorDims.mkString(",") +"\n" +
    "embedCols:" + embedCols.mkString(",") +"\n" +
    "embedInDims:" + embedInDims.mkString(",") +"\n" +
    "embedOutDims:" + embedOutDims.mkString(",") +"\n" +
    "continuousCols:" + continuousCols.mkString(",") +"\n" +
    "label:" + label

  }
}

/**
 * The Wide and Deep model used for recommendation.
 *
 * @param modelType      String. "wide", "deep", "wide_n_deep" are supported.
 *                       Default is "wide_n_deep".
 * @param numClasses     The number of classes. Positive integer.
 * @param wideBaseDims   Dimensions of wideBaseCols. The dimensions of the data in wideBaseCols
 *                       should be within the range of wideBaseDims.
 * @param wideCrossDims  Dimensions of crossed columns. The dimensions of the data in wideCrossCols
 *                       should be within the range of wideCrossDims.
 * @param indicatorDims  Dimensions of indicatorCols. The dimensions of the data in indicatorCols
 *                       should be within the range of indicatorDims.
 * @param embedInDims    Input dimension of the data in embedCols. The dimensions of the data
 *                       in embedCols should be within the range of embedInDims.
 * @param embedOutDims   The dimensions of embeddings.
 * @param continuousCols Data of continuousCols is treated as continuous values for the deep model.
 * @param hiddenLayers   Units of hidden layers for the deep model. Array of positive integers.
 *                       Default is Array(40, 20, 10).
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class WideAndDeep[T: ClassTag](
      val modelType: String = "wide_n_deep",
      val numClasses: Int,
      val wideBaseDims: Array[Int] = Array[Int](),
      val wideCrossDims: Array[Int] = Array[Int](),
      val indicatorDims: Array[Int] = Array[Int](),
      val embedInDims: Array[Int] = Array[Int](),
      val embedOutDims: Array[Int] = Array[Int](),
      val continuousCols: Array[String] = Array[String](),
      val hiddenLayers: Array[Int] = Array(40, 20, 10))(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val inputWide: ModuleNode[T] = Input(inputShape = Shape(wideBaseDims.sum + wideCrossDims.sum))
    val inputInd: ModuleNode[T] = Input(inputShape = Shape(indicatorDims.sum))
    val inputEmb = Input(inputShape = Shape(embedInDims.length))
    val inputCon = Input(inputShape = Shape(continuousCols.length))

    val wideLinear = SparseDense(numClasses).inputs(inputWide)

    modelType match {
      case "wide" =>
        val out = Activation("softmax").inputs(wideLinear)
        val model: Model[T] = Model(Array(inputWide), out)
          model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]

      case "deep" =>
        val (inputDeep, mergeList) = deepMerge(inputInd, inputEmb, inputCon)
        val deepLinear = deepHidden(mergeList.toList)
        val out = Activation("softmax").inputs(deepLinear)
        Model(inputDeep, out).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]

      case "wide_n_deep" =>
        val (inputDeep, mergeList) = deepMerge(inputInd, inputEmb, inputCon)
        val deepLinear = deepHidden(mergeList)
        val merged = Merge.merge(List(wideLinear, deepLinear), "sum")
        val out = Activation("softmax").inputs(merged)
        Model(Array(inputWide) ++ inputDeep, out)
          .asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }

  private def deepHidden(mergeList: List[ModuleNode[T]]): ModuleNode[T] = {

    val merge1 = if (mergeList.size == 1) mergeList(0) else Merge.merge(mergeList, "concat")

    var linear = Dense(hiddenLayers(0), activation = "relu").inputs(merge1)

    (1 to hiddenLayers.length - 1).map(i => {
      linear = Dense(hiddenLayers(i), activation = "relu").inputs(linear)
    })

    Dense(numClasses).inputs(linear)

  }

  private def deepMerge(ind: ModuleNode[T], emb: ModuleNode[T], cont: ModuleNode[T]) :
  (Array[ModuleNode[T]], List[ModuleNode[T]]) = {

    var embedWidth = 0
    val embLookupList = (0 to embedInDims.length - 1).map(i => {
      val lookupTable = Embedding[T](embedInDims(i) + 1, embedOutDims(i), init = "normal")
      val select = Select[T](1, embedWidth).inputs(emb)
      embedWidth = embedWidth + 1
      Flatten().inputs(lookupTable.inputs(Flatten().inputs(select)))
    }).toList

    val (input, mergeList) =
      (indicatorDims.length > 0, embedInDims.length > 0, continuousCols.length > 0) match {
      case (true, true, true) =>
        (Array(ind, emb, cont), List(ind) ++ embLookupList ++ List(cont))
      case (false, true, true) =>
        (Array(emb, cont), embLookupList ++ List(cont))
      case (true, false, true) =>
        (Array(ind, cont), List(ind) ++ List(cont))
      case (true, true, false) =>
        (Array(ind, emb), List(ind) ++ embLookupList)
      case (false, true, false) =>
        (Array(emb), embLookupList)
      case (false, false, true) =>
        (Array(cont), List(cont))
      case (true, false, false) =>
        (Array(ind), List(ind))
    }
    (input, mergeList)
  }


  /**
   * Build WideAndDeep sequential model
   *
   * @return WideAndDeep sequential model
   */
  def buildSequentialModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val fullModel = Sequential[T]()

    val wideModel = Sequential().setName("widePart")

    wideModel.add(LookupTableSparse[T](
      wideBaseDims.sum + wideCrossDims.sum, numClasses).setInitMethod(Zeros))
      .add(com.intel.analytics.bigdl.nn.CAdd(Array(numClasses)))

    // deep model
    val deepModel = Sequential[T]().setName("deepPart")
    val deepColumn = Concat[T](2)

    // add indicator encoded columns
    var indicatorWidth = 0
    if (indicatorDims.length > 0) {
      indicatorWidth = indicatorDims.sum
      deepColumn.add(Sequential[T]().add(com.intel.analytics.bigdl.nn.Narrow[T](2,
        1, indicatorWidth)))
        .setName("indicator")
    }

    // add embedding columns
    require(embedInDims.length == embedOutDims.length, s"size of embedingColumns should match")
    var embedWidth = 0
    var embedOutputDimSum = 0
    if (embedInDims.length > 0) {
      embedOutputDimSum = embedOutDims.sum
      for (i <- 0 until embedInDims.length) {
        val lookupTable = LookupTable[T](embedInDims(i), embedOutDims(i))
        lookupTable.setWeightsBias(
          Array(Tensor(embedInDims(i), embedOutDims(i)).randn(0, 0.1)))
        deepColumn.add(
          Sequential[T]().add(com.intel.analytics.bigdl.nn.Select[T](2
            , indicatorWidth + 1 + embedWidth))
            .add(lookupTable))
        embedWidth += 1
      }
    }

    // add continuous columns
    var continuousWidth = 0
    if (continuousCols.length > 0) {
      continuousWidth = continuousCols.length
      deepColumn.add(
        Sequential[T]()
          .add(com.intel.analytics.bigdl.nn.Narrow[T](2, indicatorWidth + embedWidth + 1
            , continuousWidth)))
    }

    // add hidden layers
    deepModel.add(deepColumn)
      .add(Linear[T](indicatorWidth + embedOutputDimSum + continuousWidth, hiddenLayers(0)))
      .add(ReLU())
    for (i <- 1 to hiddenLayers.length - 1) {
      deepModel.add(Linear[T](hiddenLayers(i - 1), hiddenLayers(i))).add(ReLU())
    }
    deepModel.add(Linear[T](hiddenLayers.last, numClasses))

    // create whole model
    modelType match {
      case "wide_n_deep" =>
        fullModel.add(ParallelTable().add(wideModel).add(deepModel))
          .add(CAddTable()).add(LogSoftMax())
        fullModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case "wide" =>
        wideModel.add(LogSoftMax())
        wideModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case "deep" =>
        deepModel.add(LogSoftMax())
        deepModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }
}

object WideAndDeep {
  /**
   * The factory method to create a WideAndDeep instance.
   *
   * @param modelType    String. "wide", "deep", "wide_n_deep" are supported.
   *                     Default is "wide_n_deep".
   * @param numClasses   The number of classes. Positive integer.
   * @param columnInfo   An instance of ColumnFeatureInfo.
   * @param hiddenLayers Units of hidden layers for the deep model. Array of positive integers.
   *                     Default is Array(40, 20, 10).
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      modelType: String = "wide_n_deep",
      numClasses: Int,
      columnInfo: ColumnFeatureInfo,
      hiddenLayers: Array[Int] = Array(40, 20, 10))
     (implicit ev: TensorNumeric[T]): WideAndDeep[T] = {
    require(columnInfo.wideBaseCols.length == columnInfo.wideBaseDims.length,
      s"size of wideBaseColumns should match")
    require(columnInfo.wideCrossCols.length == columnInfo.wideCrossDims.length,
      s"size of wideCrossColumns should match")
    require(columnInfo.indicatorCols.length == columnInfo.indicatorDims.length,
      s"size of indicatorColumns should match")
    require(columnInfo.embedCols.length == columnInfo.embedInDims.length &&
      columnInfo.embedCols.length == columnInfo.embedOutDims.length,
      s"size of embeddingColumns should match")

    new WideAndDeep[T](modelType,
      numClasses,
      columnInfo.wideBaseDims,
      columnInfo.wideCrossDims,
      columnInfo.indicatorDims,
      columnInfo.embedInDims,
      columnInfo.embedOutDims,
      columnInfo.continuousCols,
      hiddenLayers).build()
  }

  /**
   * The factory method to create a WideAndDeep Sequential Model instance.
   *
   * @param modelType String. "wide", "deep", "wide_n_deep" are supported.
   *                  Default is "wide_n_deep".
   * @param numClasses The number of classes. Positive integer.
   * @param columnInfo An instance of ColumnFeatureInfo.
   * @param hiddenLayers Units of hidden layers for the deep model. Array of positive integers.
   *                     Default is Array(40, 20, 10).
   */
  def sequential[@specialized(Float, Double) T: ClassTag](
                                                      modelType: String = "wide_n_deep",
                                                      numClasses: Int,
                                                      columnInfo: ColumnFeatureInfo,
                                                      hiddenLayers: Array[Int] = Array(40, 20, 10))
                                                    (implicit ev: TensorNumeric[T]): Module[T] = {
    require(columnInfo.wideBaseCols.length == columnInfo.wideBaseDims.length,
      s"size of wideBaseColumns should match")
    require(columnInfo.wideCrossCols.length == columnInfo.wideCrossDims.length,
      s"size of wideCrossColumns should match")
    require(columnInfo.indicatorCols.length == columnInfo.indicatorDims.length,
      s"size of indicatorColumns should match")
    require(columnInfo.embedCols.length == columnInfo.embedInDims.length &&
      columnInfo.embedCols.length == columnInfo.embedOutDims.length,
      s"size of embeddingColumns should match")

    new WideAndDeep[T](modelType,
      numClasses,
      columnInfo.wideBaseDims,
      columnInfo.wideCrossDims,
      columnInfo.indicatorDims,
      columnInfo.embedInDims,
      columnInfo.embedOutDims,
      columnInfo.continuousCols,
      hiddenLayers).buildSequentialModel()
  }

  /**
   * Load an existing WideAndDeep model (with weights).
   *
   * @param path       The path for the pre-defined model.
   *                   Local file system, HDFS and Amazon S3 are supported.
   *                   HDFS path should be like "hdfs://[host]:[port]/xxx".
   *                   Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](
      path: String,
      weightPath: String = null)(implicit ev: TensorNumeric[T]): WideAndDeep[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[WideAndDeep[T]]
  }
}
