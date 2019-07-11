package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Mean
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad

import scala.reflect.ClassTag

class InternalAttn[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
//    var w = AutoGrad.mm(q, k) // w: (batch, nHead, seqLen, seqLen)
//    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)
//
//    if (!bidirectional) {
//      w = w * maskValue + (maskValue * (-1) + 1) * -1e9
//    }
//
//    if (attention_mask != null) {
//      w = w + attention_mask
//    }
//    q, v:(batch, nHead, seqLen, hiddenSize/nHead)
    // k:(batch, nHead, hiddenSize/nHead, seqLen)
    val q = input[Tensor[T]](1)
    val qSize = q.size()
    val qN = q.view(Array(qSize(0)*qSize(1), qSize(2), qSize(3)))
    val k = input[Tensor[T]](2)
    val kSize = k.size()
    val kN = k.view(Array(kSize(0)*kSize(1), kSize(2), kSize(3)))
//    val v = input[Tensor[T]](3)
//    val vSize = v.size()
//    val vN = v.view(Array(vSize(0)*vSize(1), vSize(2), vSize(3)))
    val attn_mask = input[Tensor[T]](3)
    val attnSize = attn_mask.size()

//    /** res_i = res_i + (alpha * batch1_i * batch2_i) */
//    def baddbmm(alpha: T, batch1: Tensor[T], batch2: Tensor[T]): Tensor[T]

    val t = ev.fromType(scala.math.sqrt(qSize(3)))
    val outputSize = Array(qSize(0), qSize(1), qSize(2), qSize(2))
//    output = attn_mask.view(attnSize(0)*attnSize(1), attnSize(2), attnSize(3)).expand(outputSize).contiguous()
//    output.baddbmm(t, qN, kN)
    output = Tensor[T](outputSize).view(Array(qSize(0)*qSize(1), qSize(2), qSize(2)))
    output.baddbmm(t, qN, kN)

    output = output.view(outputSize)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = input
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Tensor[T]): Unit = {
  }
}
