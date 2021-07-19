package org.apache.spark.ml.bundle.ops.feature

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.zoo.pipeline.nnframes.NNClassifierModel
import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.OpModel
import org.apache.spark.ml.bundle.{ParamSpec, SimpleParamSpec, SimpleSparkOp, SparkBundleContext}

class NNClassifierOp extends SimpleSparkOp[NNClassifierModel[Float]] {
  override val Model: OpModel[SparkBundleContext, NNClassifierModel[Float]] = new OpModel[SparkBundleContext, NNClassifierModel[Float]] {
    override val klazz: Class[NNClassifierModel[Float]] = classOf[NNClassifierModel[Float]]

    override def opName: String = "NNClassifierModel"

    override def store(model: Model, obj: NNClassifierModel[Float])
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
      val oos = new ObjectOutputStream(stream)
      oos.writeObject(obj.model)
      oos.close

      model.withValue("bytes", Value.byteList(stream.toByteArray))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): NNClassifierModel[Float]
    = {
      val bytes = model.value("bytes").getByteList.toArray

      val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
      val value = ois.readObject
      ois.close
      NNClassifierModel(value.asInstanceOf[Module[Float]])
    }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: NNClassifierModel[Float]):
  NNClassifierModel[Float] = {
    NNClassifierModel(model.model.cloneModule())
  }

  override def sparkInputs(obj: NNClassifierModel[Float]): Seq[ParamSpec] = {
    Seq("features" -> obj.featuresCol)
  }

  override def sparkOutputs(obj: NNClassifierModel[Float]): Seq[SimpleParamSpec] = {
    Seq("prediction" -> obj.predictionCol)
  }
}
