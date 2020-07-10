#!/usr/bin/env bash


EXAMPLE_CONDA_NAME=orca-horovod-example

clear_up () {
   conda env remove ${EXAMPLE_CONDA_NAME}
}
#if image exist this two dependency, remove below
execute_or_exit(){
    echo "start example $1"
    start=$(date "+%s")
    python $2
    exit_status=$?
    if [ $exit_status -ne 0 ];
    then
        clear_up
        echo "$1 failed"
        exit $exit_status
    fi
    now=$(date "+%s")
    return $((now-start))
}

bash ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/setup_conda_env.sh
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "setup conda env failed"
    exit $exit_status
fi

execute_or_exit horovod-runner ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/simple_horovod_pytorch.py \
                                 --hadoop_conf $HADOOP_CONF \
                                 --slave_num 4 \
                                 --conda_name ${EXAMPLE_CONDA_NAME} \
                                 --executor_cores 4 \
                                 --executor_memory 2g \
                                 --extra_executor_memory_for_ray 2g
time1=$?

execute_or_exit pytorch-horovod-estimator ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/pytorch_estimator.py \
                                 --hadoop_conf $HADOOP_CONF \
                                 --slave_num 4 \
                                 --conda_name ${EXAMPLE_CONDA_NAME} \
                                 --executor_cores 4 \
                                 --executor_memory 2g \
                                 --extra_executor_memory_for_ray 2g
time2=$?


echo "#1 horovod-runner time used:$time1 seconds"
echo "#2 horovod-estimator time used:$time2 seconds"

clear_up
