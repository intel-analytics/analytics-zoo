package com.intel.analytics.zoo.models.inferenceModel

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.scalatest.FunSuite

class TFSaveModel extends FunSuite{

  test("tfnet load model"){

    val path = "/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/ncf23"
    //self built
    val inputs:Array[String] = null
    val outputs:Array[String] = null

    val model = TFNet.fromSavedModel(path, inputs, outputs)

    val input1 = Tensor[Float](1000,1)
    (1 to 1000).map(indx =>  input1.setValue(indx,1,10.0f))

    val input2 = Tensor[Float](Array[Float](10.0f), Array(1, 1))

    println(input1.isContiguous())
    val tensor = T(input2, input2)

    println(tensor)

    val output =  model.forward(tensor)
    println(output)

  }

  test("tfnet load model tf"){

    val path = "/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/ncf"
    val inputs:Array[String] = null
    val outputs:Array[String] = null

    val model = TFNet.fromSavedModel(path, inputs, outputs)

    val input1 = Tensor[Float](1000,1)
    (1 to 1000).map(indx =>  input1.setValue(indx,1, indx.toFloat))

    val input2 = Tensor[Float](Array[Float](1000.0f), Array(1, 1))

    val tensor1 = T(input1, input1,input1,input1,input1)
    val tensor2 = T(input2, input2,input2,input2,input2)

    (1 to 1000).map{ i=>
      val one =  Tensor[Float](Array[Float](i.toFloat), Array(1, 1))
      val input = T(one, one, one, one, one)
      val output = model.forward(input)
      println(output)
    }

  }

  test("tfnet load model from valinor"){

    val path = "/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/tf_ncf"
    //self built
    val inputs:Array[String] = null
    val outputs:Array[String] = null

    val model = TFNet.fromSavedModel(path, inputs, outputs)

    val input1 = Tensor[Float](1000,1)
    (1 to 1000).map(indx =>  input1.setValue(indx,1,10.0f))

    val input2 = Tensor[Float](Array[Float](10.0f), Array(1, 1))

    println(input1.isContiguous())
   //val tensor = T(input1, input1, input1, input1,input1)
    val tensor = T(input2, input2, input2, input2,input2)

    println(tensor)

    val output =  model.forward(tensor)
    println(output)

  }

}
