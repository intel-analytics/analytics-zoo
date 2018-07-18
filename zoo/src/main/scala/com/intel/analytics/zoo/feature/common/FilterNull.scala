package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


/**
 * filter null out of the iterator
 */
class FilterNull[F, T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Preprocessing[F, F] {

  override def apply(prev: Iterator[F]): Iterator[F] = {
    prev.filter( _ != null)
  }
}

object FilterNull {
  def apply[F, T: ClassTag]()(implicit ev: TensorNumeric[T]): FilterNull[F, T] =
    new FilterNull[F, T]()
}

