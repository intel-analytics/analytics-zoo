# deep_speech_tutorial

### Pre-setups

BigDL 0.2.0 with Spark 2.0.2 Release download [link](https://bigdl-project.github.io/0.1.1/#release-download/)

Spark 2.0.2 pre-built with Hadoop2.6 download [link](https://spark.apache.org/downloads.html)

Set the path where the downloads are to be the SPARK_HOME and BigDL_HOME in your start_notebook.sh. 

You can find two assembly files with extension .zip and .jar in your ${BigDL_HOME}/lib folder.

make sure PYTHON_API_ZIP_PATH and BigDL_JAR_PATH in the start_notebook.sh point to the correct path.

Also, Use command `mvn clean package` at your deepspeech project [dir](https://github.com/intel-analytics/analytics-zoo/tree/master/pipeline/deepspeech2) to create deepspeech2-0.1-SNAPSHOT-jar-with-dependencies.jar. Change the DEEPSPEECH_API variable to link with the correct corresponding one.

#### ds2 model (~387MB):
Download [ds2 model](https://drive.google.com/open?id=0B_s7AwBOnuD-ckRqQWM3WFctZmM) in the same directory with the DeepSpeech2_inference_tutorial.ipynb

