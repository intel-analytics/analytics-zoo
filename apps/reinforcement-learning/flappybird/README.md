# Deep Q Learning
This example demonstrates how to use the Deep-Q Learning algorithm with Analytics Zoo Keras-Style API together to play FlappyBird, at each state the agent will evaluate the q value of each action predicted by DQN, then decide the best action to execute (Flap or Not Flap). After 300 million time steps training, the agent can always pass the pipe and not fail. This example provides both python and notebook implementations.


# Environment
* Python2.7
* pygame
* scikit-image
* Imageio
* 	Apache Spark 1.6.0
* Scala 2.11
* Jupyter Notebook 4.1
* Analytics Zoo 0.1.0


# Run python script
* Download Analytics Zoo and build it.
* Run export SPARK_HOME=the root directory of Spark.
* Run export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project.
* if you want to train the DQN
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-cores 32 \
    --driver-memory 180g \
    --total-executor-cores 32\
    --executor-cores  8 \
    --executor-memory 180g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/flappybird/flappybird_qlearning.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_HOME}/pyzoo/zoo/examples/flappybird/flappybird_qlearning.py \
    -m "Train"
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    $*
* if you want to run a pretrained model,change the -m "Train" to -m "Run"
 

# Run with Jupyter
* Download Analytics Zoo and build it.
* Run export SPARK_HOME=the root directory of Spark.
* Run export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie MASTER = local[physcial_core_number].(Note: In case of out of memory error you'd better set a large executor-memory)
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 12g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 12g







