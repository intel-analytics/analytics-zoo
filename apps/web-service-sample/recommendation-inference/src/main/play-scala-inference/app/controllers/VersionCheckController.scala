package controllers

import java.text.SimpleDateFormat

import javax.inject._
import models.{LoadModel, RnnParams, WndParams}
import play.api.libs.json._
import play.api.mvc._

/**
  * This controller creates an `Action` to handle healthcheck.
  */

@Singleton
class VersionCheckController @Inject()(cc: ControllerComponents)
  extends AbstractController(cc) with LoadModel {

  /**
    * Create an Action to return @VersionCheckController status.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `GET` request with
    * a path of `/VersionCheck`.
    */

  def index = Action {
    val df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
    mType match {
      case "recrnn" =>
        val rnnParams = params.asInstanceOf[RnnParams]
        val modelEpoch = rnnParams.recModelVersion.get
        val modelVersion = df.format(modelEpoch)
        val skuIndexerModelEpoch = rnnParams.skuIndexerModelVersion.get
        val skuIndexerModelVersion = df.format(skuIndexerModelEpoch)

        Ok(
          Json.obj(
            "status" -> "ok",
            "recModelVersion" -> modelVersion,
            "skuIndexerModelVersion" -> skuIndexerModelVersion
          )
        )
      case "wnd" =>
        val wndParams = params.asInstanceOf[WndParams]
        val df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
        val modelEpoch = wndParams.wndModelVersion.get
        val modelVersion = df.format(modelEpoch)
        val userIndexerModelEpoch = wndParams.userIndexerModelVersion.get
        val userIndexerModelVersion = df.format(userIndexerModelEpoch)
        val itemIndexerModelEpoch = wndParams.itemIndexerModelVersion.get
        val itemIndexerModelVersion = df.format(itemIndexerModelEpoch)

        Ok(
          Json.obj(
            "status" -> "ok",
            "wndModelVersion" -> modelVersion,
            "userIndexerModelVersion" -> userIndexerModelVersion,
            "itemIndexerModelVersion" -> itemIndexerModelVersion
          )
        )
    }

  }
}
