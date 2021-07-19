package models

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.intel.analytics.zoo.models.recommendation.ColumnFeatureInfo

import scala.io.BufferedSource

trait LoadModel {
  val jsonMapper = new ObjectMapper
  jsonMapper.registerModule(DefaultScalaModule)
  val yamlMapper = new ObjectMapper(new YAMLFactory())
  yamlMapper.registerModule(DefaultScalaModule)

  val numPredicts = 20
  val configYaml: BufferedSource = ModelParams.loadConfig("./conf/modelConfig").get
  val config: Map[String, String] = yamlMapper.readValue(configYaml.reader(), classOf[Map[String, String]])
  val mType: String = config("type")

  var params: Models = mType match {
    case "recrnn" =>
      ModelParams(
        config("modelPath"),
        config("transformerPath"),
        config("lookupPath"),
        scala.util.Properties.envOrElse("configEnvironmewnt", "dev")
      )
    case "wnd" =>
      ModelParams(
        config("modelPath"),
        config("useIndexerPath"),
        config("itemIndexerPath"),
        config("lookupPath"),
        scala.util.Properties.envOrElse("configEnvironmewnt", "dev")
      )
  }

  val localColumnInfo = ColumnFeatureInfo(
    wideBaseCols = Array("loyalty_ind", "hvb_flg", "agent_smb_flg", "customer_type_nm", "sales_flg", "atcSKU", "GENDER_CD"),
    wideBaseDims = Array(2, 2, 2, 3, 2, 10, 3),
    wideCrossCols = Array("loyalty-ct"),
    wideCrossDims = Array(100),
    indicatorCols = Array("customer_type_nm", "GENDER_CD"),
    indicatorDims = Array(3, 3),
    embedCols = Array("userId", "itemId"),
    embedInDims = Array(10, 10),
    embedOutDims = Array(20, 11),
    continuousCols = Array("interval_avg_day_cnt", "STAR_RATING_AVG", "reviews_cnt")
  )
}
