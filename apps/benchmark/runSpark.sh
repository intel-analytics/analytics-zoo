#!/usr/bin/bash

# debug flag
#set -x

HYPER=`lscpu |grep "Thread(s) per core"|awk '{print $4}'`
CORES=$(($(nproc) / HYPER))

if [[ -z "${KMP_AFFINITY}" ]]; then
  KMP_AFFINITY=granularity=fine,compact
  if [[HYPER==2]]; then
    KMP_AFFINITY=${KMP_AFFINITY},1,0
  fi
fi
echo "Core number ${CORES}"

#export LD_LIBRARY_PATH=./openvino

# export SPARK_HOME=./spark-2.3.3-bin-hadoop2.7
export ANALYTICS_ZOO_HOME=~/zoo-bin

export MASTER="spark://localhost:7077"

#export OMP_NUM_THREADS=${CORES}
export KMP_BLOCKTIME=20

ITER=100
NUM_EXECUTORS=1

usage()
{
    echo "usage:
       1. Model path, e.g., *.model, *.xml
       2. Batch Size, e.g., 32
       3. Iteration, optional, default 100
       4. Numer of executors, optional, default 1
       as parameters in order. More concretely, you can run this command:
       bash runSpark.sh \\
            openvinomodel.xml \\
            64 \\
            100 \\
	          2
            "
    exit 1
}

if [ "$#" -lt 2 ]
then
    usage
else
    MODEL="$1"
    BS="$2"
fi

if [ -n "$3" ]
then
    ITER="$3"
fi

if [ -n "$4" ]
then
    NUM_EXECUTORS="$4"
fi

if [ "$#" -gt 4 ]
then
    PARAMS="$5"
fi

export ZOO_NUM_MKLTHREADS=$((CORES/NUM_EXECUTORS))

CLASS=com.intel.analytics.zoo.benchmark.inference.OpenVINOSparkPerf

# for maven
JAR=target/benchmark-0.1.0-SNAPSHOT-jar-with-dependencies.jar

${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh \
  --master ${MASTER} \
  --driver-memory 20g \
  --executor-memory 40g \
  --num-executors ${NUM_EXECUTORS} \
  --executor-cores $((CORES/NUM_EXECUTORS)) \
  --total-executor-cores ${CORES} \
  --conf spark.rpc.message.maxSize=2047
  --class ${CLASS} ${JAR} \
  -m ${MODEL} --iteration ${ITER} --batchSize ${BS} -n ${NUM_EXECUTORS}
