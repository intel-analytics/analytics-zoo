package controllers

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.tensor.Tensor
import javax.inject._
import models.{LoadModel, RnnParams, WndParams}
import play.api.libs.json._
import play.api.mvc._
import utilities.Helper._

import scala.collection.immutable.Map

/**
  * This controller creates an `Action` to handle recommendation prediction.
  */

@Singleton
class ModelController @Inject()(cc: ControllerComponents) extends AbstractController(cc) with LoadModel {

  /**
    * Create an Action to return @ModelController status.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `POST` request with
    * a path of `/recModel`.
    */

  def inference: Action[JsValue] = Action(parse.json) { request =>
    try {
      val requestJson = request.body.toString()
      val requestMap = jsonMapper.readValue(requestJson, classOf[Map[String, Any]])
      val predictionJson = mType match {
        case "recrnn" =>
          val rnnParams = params.asInstanceOf[RnnParams]
          val sku = requestMap("SKU_NUM").asInstanceOf[List[String]]
          val skuIndex = sku.map(x => leapTransform(
            requestString = x, inputCol = "SKU_NUM", outputCol = "SKU_INDEX", transformer = rnnParams.skuIndexerModel.get, mapper = jsonMapper
          ).toFloat).reverse.padTo(10, 0f).reverse

          val inputSample = Array(Sample(Tensor(skuIndex.toArray, Array(10))))

          val predict = rnnParams.recModel match {
            case Some(_) =>
              Map(
                "0" -> Map(
                  "atc" -> "true",
                  "heading" -> "Office Depot top sellers",
                  "name" -> "content2_json",
                  "type" -> "json",
                  "sku" -> rnnParams.recModel.get.predict(inputSample)
                    .map { x =>
                      val _output = x.toTensor[Float]
                      val indices = _output.topk(numPredicts, 1, false)
                      val predict = (1 to numPredicts).map{ i =>
                        val predict = indices._2.valueAt(i).toInt - 1
                        //                      val probability = Math.exp(_output.valueAt(predict).toDouble)
                        Map(
                          i -> Map(
                            "id" -> revertStringIndex(predict.toString),
                            "url" -> "N/A",
                            "categoryID" -> "N/A",
                            "categoryName" -> "N/A"
                          )
                        )
                      }
                      predict.toArray
                    }.head
                )
              )
            case None => Map("predict" -> "N/A", "probability" -> 0.0)
          }
          jsonMapper.writeValueAsString(predict)
        case "wnd" =>
          val wndParams = params.asInstanceOf[WndParams]
          val requestJson = request.body.toString()
          val requestMap = jsonMapper.readValue(requestJson, classOf[Map[String, String]])
          val sku = requestMap("COOKIE_ID")
          val atc = requestMap("SKU_NUM")
          val uid = leapTransform(sku, "COOKIE_ID", "userId", wndParams.userIndexerModel.get, jsonMapper)
          val iid = leapTransform(atc, "SKU_NUM", "itemId", wndParams.itemIndexerModel.get, jsonMapper)

          println(Files.exists(Paths.get("./modelFiles/userIndexer.zip")))

          val requestMap2 = requestMap + ("userId" -> uid.toInt, "itemId" -> iid.toInt)
          println(requestMap2)

          val (joined, atcMap) = assemblyFeature(requestMap2, wndParams.atcArray.get, localColumnInfo, 100)

          val train = createUserItemFeatureMap(joined.toArray.asInstanceOf[Array[Map[String, Any]]], localColumnInfo, "wide_n_deep")
          val trainSample = train.map(x => x.sample)
          println("Sample is created, ready to predict")

          val localPredictor = wndParams.wndModel.get
          val prediction = localPredictor.predict(trainSample)
            .zipWithIndex.map( p => {
            val id = p._2.toString
            val _output = p._1.toTensor[Float]
            val predict: Int = _output.max(1)._2.valueAt(1).toInt
            val probability = Math.exp(_output.valueAt(predict).toDouble)
            Map("predict" -> predict, "probability" -> probability, "id" -> id)
          })

          val prediction1 = prediction.map( x => {
            val result = x.map{ case (k ,v) => (k, (v, atcMap.getOrElse(v.toString, "")))}
            val predict = result("predict")._1
            val probability = result("probability")._1
            val atcSku = result("id")._2
            Map("predict" -> predict, "probability" -> probability, "atcSku" -> atcSku)
          })
          jsonMapper.writeValueAsString(prediction1)
      }
      Ok(Json.parse(predictionJson.toString))
    }

    catch{
      case _:Exception => BadRequest("Nah nah nah nah nah...this request contains bad characters...")
    }
  }
}
