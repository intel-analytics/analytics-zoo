package controllers

import javax.inject._
import play.api.libs.json._
import play.api.mvc._

/**
  * This controller creates an `Action` to handle healthcheck.
  */

@Singleton
class HomeController @Inject()(cc: ControllerComponents)
  extends AbstractController(cc) {

  /**
    * Create an Action to return healthcheck status.
    * The configuration in the `routes` file means that this method
    * will be called when the application receives a `GET` request with
    * a path of `/`.
    */
  def index = Action {
      Ok(Json.obj("status" -> "ok"))
  }
}