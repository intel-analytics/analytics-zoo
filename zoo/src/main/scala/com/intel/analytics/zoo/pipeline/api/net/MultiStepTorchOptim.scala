package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil

import scala.reflect.ClassTag

class MultiStepTorchOptim[@specialized(Float, Double) T: ClassTag](
      torchOptims: Array[TorchOptim[T]],
      trainingEpochs: Array[Int])(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
  require(torchOptims.length == trainingEpochs.length,
    s"detect you have ${torchOptims.length} optimizers, but only ${trainingEpochs.length}" +
      s" training epochs, expected ${torchOptims.length} endEpochs.")

  protected def getCurrent(epoch: Int): (TorchOptim[T], Int) = {
    var currentOptim = torchOptims(0)
    var currentEpoch = epoch
    for (i <- trainingEpochs.indices) {
      if (epoch >= trainingEpochs.slice(0, i + 1).sum) {
        if (i == trainingEpochs.length - 1) {
          currentOptim = torchOptims(i)
        } else {
          currentOptim = torchOptims(i + 1)
          currentEpoch -= trainingEpochs(i)
        }
      }
    }
    println(s"$epoch  ----------  $currentEpoch")
    (currentOptim, currentEpoch)
  }

  override def optimize(
      feval: Tensor[T] => (T, Tensor[T]),
      parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val currentEpoch = TorchOptim.getEpoch(this)
    val (currentOptim, _) = updateHyperParameter(currentEpoch)
    currentOptim.optimize(feval, parameter)
  }

  override def clearHistory(): Unit = {

  }

  override def getLearningRate(): Double = {
    val currentEpoch = TorchOptim.getEpoch(this)
    getCurrent(currentEpoch)._1.getLearningRate()
  }

  override def loadFromTable(config: Table): this.type = {
    this
  }

  protected def updateHyperParameter(epoch: Int): (TorchOptim[T], Int) = {
    val (currentOptim, currentEpoch) = getCurrent(epoch)
    val state = InternalOptimizerUtil.getStateFromOptiMethod(currentOptim)
    val driverState = InternalOptimizerUtil.getStateFromOptiMethod(currentOptim)
    state.update("epoch", currentEpoch)
    if (driverState.contains("score")) {
      state.update("score", driverState("score"))
    }
    currentOptim.updateHyperParameter()
    (currentOptim, currentEpoch)
  }

  override def updateHyperParameter(): Unit = {
    val currentEpoch = TorchOptim.getEpoch(this)
    updateHyperParameter(currentEpoch)
  }

  override def getHyperParameter(): String = {
    val currentEpoch = TorchOptim.getEpoch(this)
    getCurrent(currentEpoch)._1.getHyperParameter()
  }
}


object MultiStepTorchOptim{
  def apply[T: ClassTag](
        torchOptims: Array[TorchOptim[T]],
        epochs: Array[Int])(implicit ev: TensorNumeric[T]): MultiStepTorchOptim[T] = {
    new MultiStepTorchOptim[T](torchOptims, epochs)
  }
}
