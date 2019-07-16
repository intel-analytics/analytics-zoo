package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object SparseTensorUtils {
  def updateValues[U](tensor: Tensor[U],
    value: Array[U]): Tensor[U] = {
    Array.copy(value, 0, tensor.asInstanceOf[SparseTensor[U]]._values.array(), 0, value.length)
    tensor
  }

  def addSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    tensor2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val sparseT2 = tensor2.asInstanceOf[SparseTensor[T]]
    require(sparseT._indices.length == 2 && sparseT2._indices.length == 2,
      "Only support 2D sparse tensor")
    require(sparseT._shape.deep  == sparseT2._shape.deep,
      "added sparse tensors should have the same shape")
    val tIndice1 = sparseT._indices.head.array()
    val tIndice2 = sparseT._indices.last.array()
    val tValue = sparseT.storage().array()

    val t2Indice1 = sparseT2._indices.head.array()
    val t2Indice2 = sparseT2._indices.last.array()
    val t2Value = sparseT2.storage().array()

    val indice = ArrayBuffer[Int]()
    val indice2 = ArrayBuffer[Int]()
    val value = ArrayBuffer[T]()

    var i = 0
    var j = 0

    while (i < tIndice1.length && j < t2Indice1.length) {
      if (tIndice1(i) == t2Indice1(j)) {
        indice += tIndice1(i)
        if (tIndice2(i) == t2Indice2(j)) {
          indice2 += tIndice2(i)
          value += (ev.plus(tValue(i), t2Value(j)))
          i += 1
          j += 1
        } else if (tIndice2(i) < t2Indice2(j)) {
          indice2 += tIndice2(i)
          value += tValue(i)
          i += 1
        } else {
          indice2 += t2Indice2(j)
          value += t2Value(j)
          j += 1
        }
      } else if (tIndice1(i) > t2Indice1(j)) {
        indice += t2Indice1(j)
        indice2 += t2Indice2(j)
        value += t2Value(j)
        j += 1
      } else {
        indice += tIndice1(i)
        indice2 += tIndice2(i)
        value += tValue(i)
        i += 1
      }
    }
    if (i < tIndice1.length) {
      indice ++= tIndice1.slice(i, tIndice1.length)
      indice2 ++= tIndice2.slice(i, tIndice1.length)
      value ++= tValue.slice(i, tIndice1.length)
    } else if (j < t2Indice1.length) {
      indice ++= t2Indice1.slice(j, t2Indice1.length)
      indice2 ++= t2Indice2.slice(j, t2Indice1.length)
      value ++= t2Value.slice(j, t2Indice1.length)
    }
    SparseTensor(Array(indice.toArray, indice2.toArray), Storage[T](value.toArray),
      sparseT._shape, sparseT._shape.length)
  }

  def addDenseSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    tensor2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensor.isInstanceOf[DenseTensor[T]] && tensor2.isInstanceOf[SparseTensor[T]])
    val denseT = tensor.asInstanceOf[DenseTensor[T]]
    val dValues = denseT.storage().array()
    val kB: Int = denseT.size(2)

    val sparseT = tensor2.asInstanceOf[SparseTensor[T]]
    val rows = sparseT._indices.head.array()
    val cols = sparseT._indices.last.array()
    val values = sparseT._values.array()

    var i = 0
    while (i < rows.length) {
      dValues(rows(i) * kB + cols(i)) = ev.plus(values(i), dValues(rows(i) * kB + cols(i)))
      i += 1
    }
    tensor
  }

  def addSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Array[Tensor[T]],
    tensor2: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    require(tensor.length == tensor2.length, "tensor and tensor2 should have the same length")
    var i = 0
    val res = ArrayBuffer[Tensor[T]]()
    while (i < tensor.length) {
      res += addSparseTensor(tensor(i), tensor2(i))
      i += 1
    }
    res.toArray
  }

  def add[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    constant: T)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val row = sparseT._indices.head.array()
    val col = sparseT._indices.last.array()
    val value = sparseT._values.array()
    val res = Tensor[T](sparseT._shape).fill(constant)
    var i = 0
    while (i < sparseT.nElement()) {
      res.setValue(row(i) + 1, col(i) + 1, ev.plus(value(i), constant))
      i += 1
    }
    res
  }

  def addSparseTensorValueByConstant[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    constant: T)(implicit ev: TensorNumeric[T]): Unit = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val values = sparseT._values.array()
    var i = 0
    while (i < values.length) {
      values(i) = (ev.plus(values(i), constant))
      i += 1
    }
  }

  def dotSparseTensorValueByConstant[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    constant: T)(implicit ev: TensorNumeric[T]): Unit = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val values = sparseT._values.array()
    var i = 0
    while (i < values.length) {
      values(i) = (ev.times(values(i), constant))
      i += 1
    }
  }

  def resizeAsSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]

    SparseTensor(sparseT._shape)
  }

  def copySparseTensor[T](src: Tensor[T], dst: Tensor[T]): Unit = {
    require(src.isInstanceOf[SparseTensor[T]] && dst.isInstanceOf[SparseTensor[T]])
    val sparseSrc = src.asInstanceOf[SparseTensor[T]]
    val sparseDst = dst.asInstanceOf[SparseTensor[T]]
    require(sparseSrc._indices.head.size == sparseDst._indices.head.size &&
      sparseSrc._indices.last.size == sparseDst._indices.last.size &&
      sparseSrc._values.size == sparseDst._values.size, "dst must has the same size with src")

    sparseDst._indices.head.copy(sparseSrc._indices.head)
    sparseDst._indices.last.copy(sparseSrc._indices.last)
    sparseDst._values.copy(sparseSrc._values)
  }

  def cloneSparseTensor[T: ClassTag](src: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(src.isInstanceOf[SparseTensor[T]])
    val sparseSrc = src.asInstanceOf[SparseTensor[T]]

    SparseTensor(sparseSrc._indices.map(x => x.array().clone()),
      Storage[T](sparseSrc._values.array().clone()), sparseSrc._shape.clone())
  }

  // res = alpha * mat1 * mat2
  def mmSparseTensor[@specialized(Float, Double) T: ClassTag](
    alpha: T,
    mat1: Tensor[T],
    mat2: Tensor[T]
  )(implicit ev: TensorNumeric[T]) : Tensor[T] = {
    require(mat1.isInstanceOf[DenseTensor[T]] && mat2.isInstanceOf[SparseTensor[T]],
    "mm only support dense matrix with sparse matrix")
    val dTensor = mat1.asInstanceOf[DenseTensor[T]]
    val sTensor = mat2.asInstanceOf[SparseTensor[T]]

    val kB: Int = sTensor.size(1)
    val nB: Int = sTensor.size(2)
    val mA: Int = dTensor.size(1)
    val kA: Int = dTensor.size(2)

    val Avals = dTensor.storage().array()
    val aOffset = dTensor.storageOffset() - 1

    val Bvals = sTensor._values.array()
    val BstorageOffset = sTensor.storageOffset() - 1
    val BrowIndices = sTensor._indices(0)
    val BrowIndicesOffset = sTensor._indicesOffset(0)
    val BcolIndices = sTensor._indices(1)
    val BcolIndicesOffset = sTensor._indicesOffset(1)

    val indice = ArrayBuffer[ArrayBuffer[Int]]()
    val indice2 = ArrayBuffer[ArrayBuffer[Int]]()
    val value = ArrayBuffer[ArrayBuffer[T]]()

    // Perform matrix multiplication. The rows of sTensor are multiplied by the columns of dTensor
    var index = 0
    if (dTensor.stride(2) == 1 && dTensor.size(2) == dTensor.stride(1)) {
      while (index < sTensor.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset

        value += ArrayBuffer[T]()
        indice += ArrayBuffer[Int]()
        indice2 += ArrayBuffer[Int]()
        var n = 0
        while (n < mA) {
          value(index) +=
          ev.times(ev.times(alpha, Bvals(index + BstorageOffset)),
            Avals(n * kA + curKB + aOffset))
          indice(index) += n
          indice2(index) += curNB
          n += 1
        }
        index += 1
      }
    } else {
      while (index < sTensor.nElement()) {
        val curKB = BrowIndices(index + BstorageOffset) - BrowIndicesOffset
        val curNB = BcolIndices(index + BstorageOffset) - BcolIndicesOffset

        value += ArrayBuffer[T]()
        indice += ArrayBuffer[Int]()
        indice2 += ArrayBuffer[Int]()
        var n = 0
        while (n < mA) {
          value(index) +=
          ev.times(ev.times(alpha, Bvals(index + BstorageOffset)),
            Avals(n + mA * curKB + aOffset))
          indice(index) += n
          indice2(index) += curNB
          n += 1
        }
        index += 1
      }
    }

    var res = SparseTensor(Array(indice.head.toArray, indice2.head.toArray),
      Storage[T](value.head.toArray), Array(mA, nB), 2)
    index = 1
    while (index < value.length) {
      val s = SparseTensor(Array(indice(index).toArray, indice2(index).toArray),
        Storage[T](value(index).toArray), Array(mA, nB), 2)
      res = addSparseTensor(res, s).asInstanceOf[SparseTensor[T]]
      index += 1
    }
    res
  }

  def dotSparseTensor[@specialized(Float, Double) T: ClassTag](
    mat1: Tensor[T],
    mat2: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sparseT = mat1.asInstanceOf[SparseTensor[T]]
    val sparseT2 = mat2.asInstanceOf[SparseTensor[T]]
    require(sparseT._indices.length == 2 && sparseT2._indices.length == 2,
      "Only support 2D sparse tensor")
    require(sparseT._shape.deep  == sparseT2._shape.deep,
      "added sparse tensors should have the same shape")
    val tIndice1 = sparseT._indices.head.array()
    val tIndice2 = sparseT._indices.last.array()
    val tValue = sparseT.storage().array()

    val t2Indice1 = sparseT2._indices.head.array()
    val t2Indice2 = sparseT2._indices.last.array()
    val t2Value = sparseT2.storage().array()

    val indice = ArrayBuffer[Int]()
    val indice2 = ArrayBuffer[Int]()
    val value = ArrayBuffer[T]()

    var i = 0
    var j = 0

    while (i < tIndice1.length && j < t2Indice1.length) {
      if (tIndice1(i) == t2Indice1(j)) {
        if (tIndice2(i) == t2Indice2(j)) {
          indice += tIndice1(i)
          indice2 += tIndice2(i)
          value += (ev.times(tValue(i), t2Value(j)))
          i += 1
          j += 1
        } else if (tIndice2(i) < t2Indice2(j)) {
          i += 1
        } else {
          j += 1
        }
      } else if (tIndice1(i) > t2Indice1(j)) {
        j += 1
      } else {
        i += 1
      }
    }
    SparseTensor(Array(indice.toArray, indice2.toArray), Storage[T](value.toArray),
      sparseT._shape, sparseT._shape.length)
  }

  def divSparseTensor[@specialized(Float, Double) T: ClassTag](
    mat1: Tensor[T],
    mat2: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sparseT = mat1.asInstanceOf[SparseTensor[T]]
    val sparseT2 = mat2.asInstanceOf[SparseTensor[T]]
    require(sparseT._indices.length == 2 && sparseT2._indices.length == 2,
      "Only support 2D sparse tensor")
    require(sparseT._shape.deep  == sparseT2._shape.deep,
      "added sparse tensors should have the same shape")
    val tIndice1 = sparseT._indices.head.array()
    val tIndice2 = sparseT._indices.last.array()
    val tValue = sparseT.storage().array()

    val t2Indice1 = sparseT2._indices.head.array()
    val t2Indice2 = sparseT2._indices.last.array()
    val t2Value = sparseT2.storage().array()

    val indice = ArrayBuffer[Int]()
    val indice2 = ArrayBuffer[Int]()
    val value = ArrayBuffer[T]()

    var i = 0
    var j = 0

    while (i < tIndice1.length && j < t2Indice1.length) {
      if (tIndice1(i) == t2Indice1(j)) {
        if (tIndice2(i) == t2Indice2(j)) {
          indice += tIndice1(i)
          indice2 += tIndice2(i)
          value += (ev.divide(tValue(i), t2Value(j)))
          i += 1
          j += 1
        } else if (tIndice2(i) < t2Indice2(j)) {
          i += 1
        } else {
          j += 1
        }
      } else if (tIndice1(i) > t2Indice1(j)) {
        j += 1
      } else {
        i += 1
      }
    }
    SparseTensor(Array(indice.toArray, indice2.toArray), Storage[T](value.toArray),
      sparseT._shape, sparseT._shape.length)
  }

  def divSparseDenseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    tensor2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(tensor2.isInstanceOf[DenseTensor[T]] && tensor.isInstanceOf[SparseTensor[T]])
    val denseT = tensor2.asInstanceOf[DenseTensor[T]]
    val dValues = denseT.storage().array()
    val kB: Int = denseT.size(2)

    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val rows = sparseT._indices.head.array()
    val cols = sparseT._indices.last.array()
    val values = sparseT._values.array()

    var i = 0
    while (i < rows.length) {
      values(i) = ev.divide(values(i), dValues(rows(i) * kB + cols(i)))
      i += 1
    }
    tensor
  }

  def addcmulSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    value: T, tensor1: Tensor[T], tensor2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val t = dotSparseTensor(tensor1, tensor2)
    t.asInstanceOf[SparseTensor[T]]._values.foreach(ev.times(_, value))
    addSparseTensor(tensor, t)
  }

  def addcdivSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T],
    value: T, tensor1: Tensor[T], tensor2: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    (tensor, tensor1, tensor2)  match {
      case (a: DenseTensor[T], x: SparseTensor[T], y: DenseTensor[T]) =>
        val t = divSparseDenseTensor(x, y)
        dotSparseTensorValueByConstant(t, value)
        addDenseSparseTensor(tensor, t)
      case (a: DenseTensor[T], x: SparseTensor[T], y: SparseTensor[T]) =>
        val t = divSparseTensor(x, y)
        t.asInstanceOf[SparseTensor[T]]._values.foreach(ev.times(_, value))
        dotSparseTensorValueByConstant(t, value)
        addDenseSparseTensor(tensor, t)
      case (a: SparseTensor[T], x: SparseTensor[T], y: SparseTensor[T]) =>
        val t = divSparseTensor(x, y)
        dotSparseTensorValueByConstant(t, value)
        addSparseTensor(a, t)
      case _ =>
        throw new IllegalArgumentException(s"addcdivSparseTensor doesn't support")
    }
  }

  def sqrtSparseTensor[@specialized(Float, Double) T: ClassTag](tensor: Tensor[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    val sparseT = tensor.asInstanceOf[SparseTensor[T]]
    val values = sparseT._values.array()
    var i = 0
    while (i < values.length) {
      values(i) = ev.sqrt(values(i))
      i += 1
    }
  }
}
