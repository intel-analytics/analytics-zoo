#!/bin/bash
# start-cluster-serving
${FLINK_HOME}/bin/flink run-application \
  -c com.intel.analytics.zoo.serving.ClusterServing \
  -m  ${FLINK_UI_HOST}:${FLINK_UI_PORT} \
  --target kubernetes-application \
  -Dkubernetes.cluster-id=${RUNTIME_K8S_FLINK_CLUSER_ID} \
  -Dkubernetes.container.image=${RUNTIME_K8S_FLINK_IMAGE} \
  -Dkubernetes.service-account=${RUNTIME_K8S_FLINK_SERVICE_ACCOUNT} \
  -Dkubernetes.namespace=${RUNTIME_K8S_FLINK_NAME_SPACE} \
  -Dkubernetes.taskmanager.cpu=${TASKMANAGER_CPU_NUM} \
  -Djobmanager.memory.process.size=${JOBMANAGER_MEMORY_PROCESS_SIZE} \
  -Dtaskmanager.memory.process.size=${TASKMANAGER_MEMORY_PROCESS_SIZE} \
  -Dtaskmanager.memory.task.off-heap.size=${TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE} \
  -Dtaskmanager.memory.framework.off-heap.size=${TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE} \
  local:///opt/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar \
  --configPath ${CONFIG_DIR}/serving-conf.yaml

# start redis
${REDIS_HOME}/src/redis-server --daemonize yes --port ${REDIS_PORT} --dir ${REDIS_STORAGE} --protected-mode no --maxmemory 10g

# start http
java -jar /opt/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar \
    --servableManagerConfPath ${CONFIG_DIR}/http-conf.yaml
