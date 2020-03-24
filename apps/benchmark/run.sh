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
echo "Hardware Core number ${CORES}"

usage()
{
    echo "usage:
       1. type, tf, torch, bigdl, bigdlblas or ov, e,g., bigdl
       2. Model path, e.g., *.model, *.xml
       3. Iteration, e.g., 100
       4. Batch Size, e.g., 32
       as parameters in order. More concretely, you can run this command:
       bash run.sh \\
            bigdl \\
            /path/model \\
            100 \\
            32"
    exit 1
}


OPTIONS=""
PARM=""

if [ "$#" -lt 4 ]
then
    usage
else
    TYPE="$1"
    MODEL="$2"
    ITER="$3"
    BS="$4"
fi

case $TYPE in

  "tf" | "TF")
    echo "Analytics-Zoo with TensorFlow"
    CLASS=com.intel.analytics.zoo.benchmark.inference.TFNetPerf
    ;;

  "torch" | "TORCH")
    echo "Analytics-Zoo with PyTorch"
    CLASS=com.intel.analytics.zoo.benchmark.inference.TorchNetPerf
    export OMP_NUM_THREADS=${CORES}
    ;;

  "bigdl" | "BIGDL")
    echo "Analytics-Zoo with BigDL MKLDNN"
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLPerf
    OPTIONS='-Dbigdl.engineType=mkldnn -Dbigdl.mklNumThreads='${CORES}
    ;;

  "bigdlblas" | "BIGDLBLAS")
    echo "Analytics-Zoo with BigDL BLAS"
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLBLASPerf
    PARM="-c ${CORES}"
    ;;

  "ov" | "OV")
    echo "Analytics-Zoo with OpenVINO"
    CLASS=com.intel.analytics.zoo.benchmark.inference.OpenVINOPerf
    export OMP_NUM_THREADS=${CORES}
    export KMP_BLOCKTIME=20
    ;;

  *)
    echo "Analytics-Zoo with BigDL MKLDNN"
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLPerf
    OPTIONS='-Dbigdl.engineType=mkldnn -Dbigdl.mklNumThreads='${CORES}
    ;;
esac


# for maven
JAR=target/benchmark-0.1.0-SNAPSHOT-jar-with-dependencies.jar

java ${OPTIONS} -cp ${JAR} ${CLASS} -m ${MODEL} --iteration ${ITER} --batchSize ${BS} ${PARM}
