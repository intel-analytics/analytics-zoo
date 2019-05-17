package com.intel.analytics.zoo.optim

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag


class ZooOptimizer[T: ClassTag, D](optimzer: Optimizer[T, D],
                                   model: Module[T],
                                   trainRdd: RDD[Sample[T]],
                                   batchSize: Int,
                                   nEpochs: Int,
                                   nRdds: Int = 1) {

  def setValidation(trigger: Trigger, dataset: DataSet[MiniBatch[T]],
                    vMethods: Array[ValidationMethod[T]]): this.type = {
    optimzer.setValidation(trigger, dataset, vMethods)
    this
  }

  /**
    * Set a validate evaluation
    *
    * @param trigger             how often to evaluation validation set
    * @param sampleRDD           validate data set in type of [[RDD]] of [[Sample]]
    * @param vMethods            a set of validation method [[ValidationMethod]]
    * @param batchSize           batch size
    * @param featurePaddingParam feature padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @param labelPaddingParam   label padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @return this optimizer
    */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
                    vMethods: Array[ValidationMethod[T]], batchSize: Int,
                    featurePaddingParam: PaddingParam[T],
                    labelPaddingParam: PaddingParam[T]
                   ): this.type = {
    optimzer.setValidation(trigger, sampleRDD, vMethods, batchSize,
      featurePaddingParam, labelPaddingParam)
    this
  }

  /**
    * Set a validate evaluation
    *
    * @param trigger   how often to evaluation validation set
    * @param sampleRDD validate data set in type of [[RDD]] of [[Sample]]
    * @param vMethods  a set of validation method [[ValidationMethod]]
    * @param batchSize batch size
    * @return this optimizer
    */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
                    vMethods: Array[ValidationMethod[T]], batchSize: Int)
  : this.type = {
    optimzer.setValidation(trigger, sampleRDD, vMethods, batchSize)
    this
  }

  /**
    * Set validate evaluation
    *
    * @param trigger   how often to evaluation validation set
    * @param sampleRDD validate data set in type of [[RDD]] of [[Sample]]
    * @param vMethods  a set of validation method [[ValidationMethod]]
    * @param batchSize batch size
    * @param miniBatch construct MiniBatch with a specified miniBatch type
    * @return
    */
  def setValidation(trigger: Trigger, sampleRDD: RDD[Sample[T]],
                    vMethods: Array[ValidationMethod[T]], batchSize: Int, miniBatch: MiniBatch[T])
  : this.type = {
    optimzer.setValidation(trigger, sampleRDD, vMethods, batchSize, miniBatch)
    this
  }

  /**
    * Set a check point saved at `path` triggered by `trigger`
    *
    * @param path    the directory to save
    * @param trigger how often to save the check point
    * @return the optimizer
    */
  def setCheckpoint(path: String, trigger: Trigger): this.type = {
    optimzer.setCheckpoint(path, trigger)
    this
  }

  /**
    * Get the directory of saving checkpoint
    */
  def getCheckpointPath(): Option[String] = {
    optimzer.getCheckpointPath()
  }

  /**
    * Enable train summary.
    */
  def setTrainSummary(trainSummary: TrainSummary): this.type = {
    optimzer.setTrainSummary(trainSummary)
    this
  }

  /**
    * Enable validation summary.
    */
  def setValidationSummary(validationSummary: ValidationSummary): this.type = {
    optimzer.setValidationSummary(validationSummary)
    this
  }

  /**
    * Enable overwrite saving checkpoint
    */
  def overWriteCheckpoint(): this.type = {
    optimzer.overWriteCheckpoint()
    this
  }

  /**
    * Set a model to the optimizer.
    * Notice: if current optimMethod in this optimizer is not a global optimMethod,
    * this setModel will throw an exception. You should use setModelAndOptimMethods instead.
    *
    * @param newModel new model
    */
  def setModel(newModel: Module[T]): this.type = {
    // check if the old optimMethods is a global one.
    optimzer.setModel(newModel)
    this
  }

  /**
    * Set new model and new optimMethods to the optimizer.
    *
    * @param newModel        new model
    * @param newOptimMethods new optimMethods
    */
  def setModelAndOptimMethods(
                               newModel: Module[T],
                               newOptimMethods: Map[String, OptimMethod[T]]): this.type = {
    // check if the old optimMethods is a global one.
    optimzer.setModelAndOptimMethods(newModel, newOptimMethods)
    this
  }


  /**
    * Set new train dataset.
    * User can supply a customized implementation of trait MiniBatch to define
    * how data is organized and retrieved in a mini batch.
    *
    * @param sampleRDD     training Samples
    * @param batchSize     mini batch size
    * @param miniBatchImpl An User-Defined MiniBatch implementation.
    * @return the Optimizer
    */
  def setTrainData(sampleRDD: RDD[Sample[T]],
                   batchSize: Int,
                   miniBatchImpl: MiniBatch[T]): this.type = {
    optimzer.setTrainData(sampleRDD, batchSize, miniBatchImpl)
    this
  }

  /**
    * Set new train dataset.
    *
    * @param sampleRDD           training Samples
    * @param batchSize           mini batch size
    * @param featurePaddingParam feature padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @param labelPaddingParam   label padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @return the optimizer
    */
  def setTrainData(sampleRDD: RDD[Sample[T]],
                   batchSize: Int,
                   featurePaddingParam: PaddingParam[T] = null,
                   labelPaddingParam: PaddingParam[T] = null): this.type = {
    optimzer.setTrainData(sampleRDD, batchSize, featurePaddingParam, labelPaddingParam)
    this
  }


  /**
    * Set a new criterion to the optimizer
    *
    * @param newCriterion new criterion
    */
  def setCriterion(newCriterion: Criterion[T]): this.type = {
    optimzer.setCriterion(newCriterion)
    this
  }


  /**
    * Set a state(learning rate, epochs...) to the optimizer
    *
    * @param state the state to be saved
    */
  def setState(state: Table): this.type = {
    optimzer.setState(state)
    this
  }

  /**
    * Set an optimization method
    *
    * @param method optimization method
    */
  def setOptimMethod(method: OptimMethod[T]): this.type = {
    optimzer.setOptimMethod(method)
    this
  }

  /**
    * Set optimization methods for each submodule.
    *
    * @param method A mapping of submodule -> OptimMethod
    */
  def setOptimMethods(method: Map[String, OptimMethod[T]]): this.type = {
    optimzer.setOptimMethods(method)
    this
  }

  /**
    * When to stop, passed in a [[Trigger]]
    *
    * @param endWhen when to end
    * @return the optimizer
    */
  def setEndWhen(endWhen: Trigger): this.type = {
    optimzer.setEndWhen(Trigger.maxEpoch(1))
    this
  }

  /**
    * Set dropping a certain percentage (`dropPercentage`) of models during distributed
    * training to accelerate, because some cached model may take too long.
    *
    * @param dropPercentage    drop percentage
    * @param maxDropPercentage max drop percentage
    * @param batchsize         batch size
    * @param warmupIteration   how may iteration to warm up
    * @return this optimizer
    */
  def setDropModuleProperty(dropPercentage: Double, maxDropPercentage: Double,
                            batchsize: Int = 100, warmupIteration: Int = 200): this.type = {
    optimzer.setDropModuleProperty(dropPercentage, maxDropPercentage, batchsize, warmupIteration)
    this
  }

  def prepareInput(): Unit = {}

  /**
    * Disable gradient clipping
    *
    * @return
    */
  def disableGradientClipping()
  : this.type = {
    optimzer.disableGradientClipping()
    this
  }

  /**
    * Set constant gradient clipping
    *
    * @param min the minimum value to clip by
    * @param max the maximum value to clip by
    * @return
    */
  def setConstantGradientClipping(min: Double, max: Double)
  : this.type = {
    optimzer.setConstantGradientClipping(min, max)
    this
  }


  /**
    * Clip gradient to a maximum L2-norm
    *
    * @param l2NormThreshold gradient L2-Norm threshold
    * @return
    */

  def setGradientClippingByl2Norm(l2NormThreshold: Double)
  : this.type = {
    optimzer.setGradientClippingByl2Norm(l2NormThreshold)
    this
  }

  def optimize() = {

    if (nRdds == 1) {
      optimzer.optimize()
    } else {

      val splits: Array[Double] = (0 to nRdds - 1).map(x => 1.0 / nRdds).toArray
      val trainRddArray: Array[RDD[Sample[T]]] = trainRdd.randomSplit(splits, 1L)

      trainRddArray.map(x => x.persist(StorageLevel.DISK_ONLY))

      val allIndex = (1 to nEpochs).flatMap(x => (0 to nRdds - 1))

      optimzer.setEndWhen(Trigger.maxEpoch(1))

      allIndex.map(index => {
        optimzer
          .setModel(model)
          .setTrainData(trainRddArray(index), batchSize)
          .optimize()
      })

    }
  }
}

object ZooOptimizer {

  /**
    * Apply an Optimizer.
    *
    * @param model               model will be optimized
    * @param sampleRDD           training Samples
    * @param criterion           loss function
    * @param batchSize           mini batch size
    * @param nRdds             splits of train rdd
    * @param featurePaddingParam feature padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @param labelPaddingParam   label padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @return An optimizer
    */
  def apply[T: ClassTag](
                          model: Module[T],
                          sampleRDD: RDD[Sample[T]],
                          criterion: Criterion[T],
                          batchSize: Int,
                          nEpochs: Int,
                          nRdds: Int = 1,
                          featurePaddingParam: PaddingParam[T] = null,
                          labelPaddingParam: PaddingParam[T] = null
                        )(implicit ev: TensorNumeric[T]): ZooOptimizer[T, MiniBatch[T]] = {

    val optimizer = Optimizer(model, sampleRDD, criterion, batchSize, featurePaddingParam, labelPaddingParam)
    new ZooOptimizer(optimizer, model, sampleRDD, batchSize, nEpochs, nRdds)
  }


  /**
    * Apply an optimizer.
    * User can supply a customized implementation of trait MiniBatch to define
    * how data is organize and retrieved in a mini batch.
    *
    * @param model         model will be optimized
    * @param sampleRDD     training Samples
    * @param criterion     loss function
    * @param batchSize     mini batch size
    * @param miniBatchImpl An User-Defined MiniBatch implementation
    * @param nSplits       splits of train rdd
    * @return an new Optimizer
    */
  def apply[T: ClassTag](
                          model: Module[T],
                          sampleRDD: RDD[Sample[T]],
                          criterion: Criterion[T],
                          batchSize: Int,
                          miniBatchImpl: MiniBatch[T],
                          nRdds: Int,
                          nSplits: Int
                        )(implicit ev: TensorNumeric[T]): ZooOptimizer[T, MiniBatch[T]] = {
    val optimizer = Optimizer(model, sampleRDD, criterion, batchSize, miniBatchImpl)
    new ZooOptimizer(optimizer, model, sampleRDD, batchSize, nRdds, nSplits)
  }

}
