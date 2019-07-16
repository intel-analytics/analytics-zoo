package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.SparseAbstractModule
import com.intel.analytics.bigdl.nn.{BigDLWrapperUtils, Container}
import com.intel.analytics.bigdl.tensor.{Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Util._
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

class ZooModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new ZooModelBroadcast[T]()
  }
}

class ZooModelBroadcast[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {
//class ZooModelBroadcast[T: ClassTag]()
//  (implicit ev: TensorNumeric[T]) extends ModelBroadcastImp[T] {

  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _
  private var broadcastSparseParameters: Broadcast[Array[Tensor[T]]] = _
  private var nodeNumber : Int = _
  private var coreNumber : Int = _

  private def setNodeAndCore(): Unit = {
    nodeNumber = Engine.nodeNumber()
    coreNumber = Engine.coreNumber()
  }

  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    println("Enter ZooModelBroadcast!")
    CachedModels.deleteAll(uuid) // delete the models on driver

    // broadcast Consts
    if (model.isInstanceOf[Container[_, _, T]]) {
      val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
      // TODO: broadcast Const, model structure and weight in the same broadcast.
      broadcastConsts = sc.broadcast(moduleConsts)
    }
    // broadcast weight and model
    val weightsBias = getAndClearWeightBias(model.parameters())
    val sparseGWeightsBias =
      BigDLWrapperUtils.getAndClearWeightBias(model.asInstanceOf[SparseAbstractModule[T]].sparseParameters()._1)
    broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
    broadcastParameters = sc.broadcast(weightsBias)
    broadcastSparseParameters = sc.broadcast(sparseGWeightsBias)

    // For quantized model if we don't clone weightsBias, the original model will be released also
    // when we delete all models used in `ModelBroadcast`.
    putWeightBias(SerializationUtils.clone(weightsBias), model)
    initGradWeightBias(weightsBias, model)

    BigDLWrapperUtils.putSparseGWeightBias(SerializationUtils.clone(sparseGWeightsBias), model)
    // TODO: check if need init sparse gradients

    setNodeAndCore()
    this
  }

  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
//    val localModel = super.value(initGradient, shareWeight)
//    val sparseParameters = if (shareWeight) {
//      broadcastSparseParameters.value
//    } else {
//      SerializationUtils.clone(broadcastSparseParameters.value)
//    }
//    BigDLWrapperUtils.putSparseGWeightBias(sparseParameters, localModel)

    Engine.setNodeAndCore(nodeNumber, coreNumber)
    CachedModels.deleteAll(uuid)

    val localModel = broadcastModel.value.model.cloneModule()
    val _uuid = broadcastModel.value.uuid
    CachedModels.add(_uuid, localModel)

    val (parameters, sparseParameters) = if (shareWeight) {
      (broadcastParameters.value, broadcastSparseParameters.value)
    } else {
      (SerializationUtils.clone(broadcastParameters.value),
        SerializationUtils.clone(broadcastSparseParameters.value))
    }

    // share weight
    putWeightBias(parameters, localModel)
    BigDLWrapperUtils.putSparseGWeightBias(sparseParameters, localModel)
    // share Consts
    if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
      putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
    }
    // init gradient
    if (initGradient) {
      initGradWeightBias(broadcastParameters.value, localModel)

      // TODO: check if need init sparse gradients
    }
    localModel
  }
}
