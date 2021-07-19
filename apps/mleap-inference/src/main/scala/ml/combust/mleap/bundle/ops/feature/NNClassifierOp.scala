package ml.combust.mleap.bundle.ops.feature

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.Module
import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl.{Model, Value}
import ml.combust.bundle.op.OpModel
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.core.feature.MLeapNNClassifierModel
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.transformer.feature.MLeapNNClassifier

class NNClassifierOp extends MleapOp[MLeapNNClassifier, MLeapNNClassifierModel]{

  override val Model: OpModel[MleapContext, MLeapNNClassifierModel] = new OpModel[MleapContext, MLeapNNClassifierModel] {
    override val klazz: Class[MLeapNNClassifierModel] = classOf[MLeapNNClassifierModel]

    override def opName: String = "NNClassifierModel"

    override def store(model: Model, obj: MLeapNNClassifierModel)
                      (implicit context: BundleContext[MleapContext]): Model = {
      val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
      val oos = new ObjectOutputStream(stream)
      oos.writeObject(obj.model)
      oos.close

      model.withValue("bytes", Value.byteList(stream.toByteArray))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[MleapContext]): MLeapNNClassifierModel = {

      val bytes = model.value("bytes").getByteList.toArray
      val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
      val value = ois.readObject
      ois.close
      MLeapNNClassifierModel(value.asInstanceOf[Module[Float]])
    }
  }

  override def model(node: MLeapNNClassifier): MLeapNNClassifierModel = node.model
}
