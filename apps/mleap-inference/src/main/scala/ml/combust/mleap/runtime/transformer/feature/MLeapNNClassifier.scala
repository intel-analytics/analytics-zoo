package ml.combust.mleap.runtime.transformer.feature

import ml.combust.mleap.core.feature.MLeapNNClassifierModel
import ml.combust.mleap.core.types.NodeShape
import ml.combust.mleap.runtime.frame.{SimpleTransformer, Transformer}
import ml.combust.mleap.runtime.function.UserDefinedFunction

case class MLeapNNClassifier(override val uid: String = Transformer.uniqueName("MLeapNNClassifierModel"),
                        override val shape: NodeShape,
                        override val model: MLeapNNClassifierModel) extends
  SimpleTransformer {
  override val exec: UserDefinedFunction = (value: Seq[Float]) => model(value): Double
}
