#!/bin/bash
#setup pathes
SPARK_HOME=/home/wickedspoon/Documents/work/spark-2.0.1-bin-hadoop2.6
BigDL_HOME=/home/wickedspoon/Documents/work/dist-spark-2.0.2-scala-2.11.8-linux64-0.2.0-dist
PYTHON_API_ZIP_PATH=${BigDL_HOME}/lib/bigdl-0.2.0-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/lib/bigdl-SPARK_2.0-0.2.0-jar-with-dependencies.jar
export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"
export DEEPSPEECH_API=/home/wickedspoon/Documents/work/analytics-zoo/pipeline/deepspeech2/target/deepspeech2-0.1-SNAPSHOT-jar-with-dependencies.jar

${SPARK_HOME}/bin/pyspark \
  --master local[4] \
  --driver-memory 4g \
  --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
  --py-files ${BigDL_JAR_PATH},${DEEPSPEECH_API} \
  --jars ${BigDL_JAR_PATH},${DEEPSPEECH_API}\
  --conf spark.driver.extraClassPath=${BigDL_JAR_PATH}:/${DEEPSPEECH_API}  \
  --conf spark.executor.extraClassPath=${BigDL_JAR_PATH}:/${DEEPSPEECH_API} 
