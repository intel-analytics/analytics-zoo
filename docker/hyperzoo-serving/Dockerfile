ARG JOBMANAGER_MEMORY_PROCESS_SIZE=15g
ARG TASKMANAGER_MEMORY_PROCESS_SIZE=15g
ARG TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE=2097152b
ARG TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE=2048mb
ARG TASKMANAGER_CPU_NUM=20.0
ARG RUNTIME_K8S_FLINK_NAME_SPACE=stream
ARG RUNTIME_K8S_FLINK_SERVICE_ACCOUNT=stream
ARG RUNTIME_K8S_FLINK_IMAGE=${RUNTIME_K8S_FLINK_IMAGE}
ARG RUNTIME_K8S_FLINK_CLUSER_ID=cluster-serving
ARG FLINK_UI_HOST=localhost
ARG FLINK_UI_PORT=8081
ARG REDIS_HOME=/opt/redis-5.0.5
ARG REDIS_STORAGE=/opt/redis-storage
ARG REDIS_PORT=36379
ARG REDIS_VERSION=5.0.5
ARG ANALYTICS_ZOO_VERSION=0.11.0-SNAPSHOT
ARG SPARK_VERSION=2.4.6
ARG BIGDL_VERSION=0.13.0
ARG SERVING_MODELS_URL=models_url

#stage.1 redis
FROM ubuntu:18.04 as redis
ARG REDIS_VERSION

RUN apt-get update --fix-missing && \
    apt-get install -y wget make gcc && \
    wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-*.tar.gz -C /opt && \
    rm redis-*.tar.gz && \
    cd /opt/redis-${REDIS_VERSION} && \
    make && \
    echo "bind 0.0.0.0" >> /opt/redis-${REDIS_VERSION}/redis.conf

#stage.2 flink
FROM flink:1.13.0-scala_2.11
ARG JOBMANAGER_MEMORY_PROCESS_SIZE
ARG TASKMANAGER_MEMORY_PROCESS_SIZE
ARG TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE
ARG TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE
ARG TASKMANAGER_CPU_NUM
ARG RUNTIME_K8S_FLINK_NAME_SPACE
ARG RUNTIME_K8S_FLINK_SERVICE_ACCOUNT
ARG RUNTIME_K8S_FLINK_IMAGE
ARG RUNTIME_K8S_FLINK_CLUSER_ID
ARG FLINK_UI_HOST
ARG FLINK_UI_PORT
ARG REDIS_HOME
ARG REDIS_STORAGE
ARG REDIS_PORT
ARG SPARK_VERSION
ARG ANALYTICS_ZOO_VERSION
ARG BIGDL_VERSION
ARG SERVING_MODELS_URL

ENV BIGDL_VERSION                               ${BIGDL_VERSION}
ENV ANALYTICS_ZOO_VERSION                       ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION                               ${SPARK_VERSION}
ENV JOBMANAGER_MEMORY_PROCESS_SIZE              ${JOBMANAGER_MEMORY_PROCESS_SIZE}
ENV TASKMANAGER_MEMORY_PROCESS_SIZE             ${TASKMANAGER_MEMORY_PROCESS_SIZE}
ENV TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE       ${TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE}
ENV TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE  ${TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE}
ENV RUNTIME_K8S_FLINK_NAME_SPACE                ${RUNTIME_K8S_FLINK_NAME_SPACE}
ENV RUNTIME_K8S_FLINK_SERVICE_ACCOUNT           ${RUNTIME_K8S_FLINK_SERVICE_ACCOUNT}
ENV RUNTIME_K8S_FLINK_IMAGE                     ${RUNTIME_K8S_FLINK_IMAGE}
ENV RUNTIME_K8S_FLINK_CLUSER_ID                 ${RUNTIME_K8S_FLINK_CLUSER_ID}
ENV FLINK_UI_HOST                               ${FLINK_UI_HOST}
ENV FLINK_UI_PORT                               ${FLINK_UI_PORT}
ENV REDIS_HOME                                  ${REDIS_HOME}
ENV REDIS_PORT                                  ${REDIS_PORT}
ENV CONFIG_DIR                                  /opt/work/conf
ENV REDIS_STORAGE                               ${REDIS_STORAGE}
ENV ANALYTICS_ZOO_VERSION                       0.11.0
ENV TF_DISABLE_MKL                              1
ENV TASKMANAGER_CPU_NUM                         ${TASKMANAGER_CPU_NUM}

COPY --from=redis ${REDIS_HOME} ${REDIS_HOME}
COPY ./download-cluster-serving-all-zip.sh /hyper-zoo/
COPY ./start*.sh /opt/work/scripts/
RUN apt-get update --fix-missing && \
    apt-get -y install apt-utils vim unzip wget && \
    mkdir ${REDIS_STORAGE} && \
    cd /hyper-zoo && \
    chmod a+x /hyper-zoo/download-cluster-serving-all-zip.sh && \
    /hyper-zoo/download-cluster-serving-all-zip.sh && \
    cp /hyper-zoo/cluster-serving/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-*jar /opt/work/ && \
    cd .. && \
    rm -rf /hyper-zoo && \
    chmod a+x /opt/work/scripts/* && \
#downloads models
    wget ${SERVING_MODELS_URL} && \
    tar -xvf ./*tgz -C /opt && \
    rm -rf ./*tgz

COPY ./*.yaml ${CONFIG_DIR}/
