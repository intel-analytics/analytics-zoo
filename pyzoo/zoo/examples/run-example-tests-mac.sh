#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=$(find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar")
export ANALYTICS_ZOO_PYZIP=$(find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip")
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_CLASSPATH=${ANALYTICS_ZOO_JAR}

set -e

echo "#14 start example test for streaming Text Classification"
if [ -d analytics-zoo-data/data/streaming/text-model ]; then
  echo "analytics-zoo-data/data/streaming/text-model already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
  unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 2g \
  --executor-memory 5g \
  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/textclassification/streaming_text_classification.py \
  --model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
  --index_path analytics-zoo-data/data/streaming/text-model/word_index.txt \
  --input_file analytics-zoo-data/data/streaming/text-model/textfile/ >1.log &
while :; do
  echo "I am strong and I am smart" >>analytics-zoo-data/data/streaming/text-model/textfile/s
  if [ -n "$(grep "top-5" 1.log)" ]; then
    echo "----Find-----"
    kill -9 $(ps -ef | grep streaming_text_classification | grep -v grep | awk '{print $2}')
    rm 1.log
    sleep 1s
    break
  fi
done
now=$(date "+%s")
time14=$((now - start))

echo "#14 streaming text classification time used: $time14 seconds"
