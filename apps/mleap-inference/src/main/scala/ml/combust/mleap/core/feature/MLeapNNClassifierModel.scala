package ml.combust.mleap.core.feature

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.common.{SeqToTensor, TensorToSample}
import ml.combust.mleap.core.Model
import ml.combust.mleap.core.types._

case class MLeapNNClassifierModel(
    @transient model: Module[Float])(implicit ev: TensorNumeric[Float]) extends Model {

  def apply(input: Seq[Float]): Double = {
    val sample = (SeqToTensor() -> TensorToSample()).apply(Iterator(input)).next()
    val output = model.forward(sample.feature()).toTensor.squeeze()
    ev.toType[Double](output.max(1)._2.valueAt(1))
  }

  override def inputSchema: StructType = StructType("features" -> ListType(BasicType.Float)).get

  override def outputSchema: StructType = StructType("prediction" -> ScalarType.Double).get

}