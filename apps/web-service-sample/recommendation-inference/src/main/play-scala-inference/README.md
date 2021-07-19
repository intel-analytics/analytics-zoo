# Zoo serving using Analytics Zoo, Mleap, and Play
A generic webservice framework to serve models trained by Analytics Zoo and Spark using Analytics Zoo, Mleap, and Play

## Requirements
```scala
scalaVersion := "2.11.12"

val jacksonVersion = "2.6.7"
val sparkVersion = "2.3.1"
val analyticsZooVersion = "0.3.0"
val mleapVersion = "0.12.0"

libraryDependencies += guice
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.7.1-spark_2.3.1" % analyticsZooVersion
//libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % jacksonVersion
libraryDependencies += "com.fasterxml.jackson.dataformat" % "jackson-dataformat-yaml" % jacksonVersion
libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.11.354"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % mleapVersion
libraryDependencies += "ml.combust.mleap" %% "mleap-spark-extension" % mleapVersion
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % jacksonVersion
dependencyOverrides += "com.google.guava" % "guava" % "17.0"
```

## How to run
#### Model config file
To launch the webservice, you need to supply a modelConfig file first (conf/modelConfig), see below example:
```yaml
# Sample WND serving config
type: wnd
modelPath: tmp/WDModel
useIndexerPath: tmp/userIndexer.zip
itemIndexerPath: tmp/itemIndexer.zip
lookupPath: tmp/ATCSKU.csv
```
Current webservice support session recommender and wide and deep model trained by Analytics Zoo

#### Use terminal
You need to have sbt installed on your machine
```bash
sbt runProd
``` 
#### Package code to one jar and run directly using the java command
```bash
sbt -J-Xmx2G assembly
java -Xmx2g -jar ${location of assembled jar}
```

## API Introduction
you can use postman to send request

### Check health
**Path:** /`healthcheck`  
**Methods:** GET  
**Params:**  
None  
**Return:**
```json
{"status": "ok"}
```

### Check model version
**Path:** /`versioncheck`  
**Methods:** GET  
**Params:**  
None  
**Return:**
```json
{
    "status": "ok",
    "recModelVersion": "2019-01-04 13:22:01.000",
    "skuIndexerModelVersion": "2019-01-04 12:49:02.000"
}
```

### Run model
#### Session recommender model
**Path:** /`inference`  
**Methods:** POST  
**Params:**  
```json
{
    "SESSION_ID": "12345",
    "SKU_NUM": ["798644", "799238", "8284111"]
}
``` 
**Return:**
```json
{
    "0": {
        "heading": "Office Depot top sellers",
        "name": "content2_json",
        "sku": [
            {
                "1": {
                    "id": "9917369",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "2": {
                    "id": "6979858",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "3": {
                    "id": "604732",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "4": {
                    "id": "2774708",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "5": {
                    "id": "6630902",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "6": {
                    "id": "4474320",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "7": {
                    "id": "776948",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "8": {
                    "id": "299388",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "9": {
                    "id": "6979871",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "10": {
                    "id": "147954",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "11": {
                    "id": "282127",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "12": {
                    "id": "178878",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "13": {
                    "id": "485681",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "14": {
                    "id": "351975",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "15": {
                    "id": "460958",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "16": {
                    "id": "470796",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "17": {
                    "id": "465617",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "18": {
                    "id": "493894",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "19": {
                    "id": "4981958",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "20": {
                    "id": "9809973",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            }
        ],
        "atc": "true",
        "type": "json"
    }
}
```  

#### Wide and deep model
**Path:** /`inference`  
**Methods:** POST  
**Params:**  
```json
{
    "COOKIE_ID": "80031540652853011880052",
    "ATC_SKU": ["677252", "440539", "429457"],
    "loyalty_ind": 1,
    "od_card_user_ind": 1,
    "hvb_flg": 1,
    "agent_smb_flg": 1,
    "interval_avg_day_cnt": 159.36,
    "customer_type_nm": "CONSUMER",
    "SKU_NUM": "244559",
    "rating": 2,
    "STAR_RATING_AVG": 1,
    "reviews_cnt": 1,
    "sales_flg": 1,
    "GENDER_CD": "F"
}
``` 
**Return:**
```json
[
    {
        "predict": 1,
        "probability": 0.7123838989001036,
        "atcSku": "677252"
    },
    {
        "predict": 1,
        "probability": 0.6807899545355797,
        "atcSku": "440539"
    },
    {
        "predict": 1,
        "probability": 0.6962240684141258,
        "atcSku": "429457"
    }
]
```  

### Update model
#### Session recommender model
**Path:** /`update`  
**Methods:** POST  
**Params:**  
```json
{
	"modelPath": "tmp/rnnModel",
	"transformerPath": "tmp/skuIndexer.zip",
	"lookupPath": "tmp/skuLookUp"
}
```  
**Return:**
```json
{
    "status": "success"
}
```
**Error_message:** ""Nah nah nah nah nah...this request contains bad characters...""

#### Wide and deep model
**Path:** /`update`  
**Methods:** POST  
**Params:**  
```json
{
	"modelPath": "tmp/WDModel",
	"useIndexerPath": "tmp/userIndexer.zip",
	"itemIndexerPath": "tmp/itemIndexer.zip",
	"lookupPath": "tmp/ATCSKU.csv"
}
```  
**Return:**
```json
{
    "status": "success"
}
```
**Error_message:** ""Nah nah nah nah nah...this request contains bad characters...""

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)


