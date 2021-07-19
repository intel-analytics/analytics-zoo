# Hadoop/YARN User Guide

Hadoop version: Hadoop >= 2.7 or CDH 5.X, Hadoop 3.X or CHD 6.X are not supported

---

You can run Analytics Zoo programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install Analytics Zoo or any Python libraries in the cluster).

### **1. Prepare Environment**

- You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment _**on the local client machine**_. Create a conda environment and install all the needed Python libraries in the created conda environment:

  ```bash
  conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
  conda activate zoo

  # Use conda or pip to install all the needed Python dependencies in the created conda environment.
  ```

- You need to download and install JDK in the environment, and properly set the environment variable `JAVA_HOME`, which is required by Spark. __JDK8__ is highly recommended.

  You may take the following commands as a reference for installing [OpenJDK](https://openjdk.java.net/install/):

  ```bash
  # For Ubuntu
  sudo apt-get install openjdk-8-jre
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

  # For CentOS
  su -c "yum install java-1.8.0-openjdk"
  export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.282.b08-1.el7_9.x86_64/jre

  export PATH=$PATH:$JAVA_HOME/bin
  java -version  # Verify the version of JDK.
  ```

- Check the Hadoop setup and configurations of your cluster. Make sure you properly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:

  ```bash
  export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
  ```

### **1.1 Setup for CDH**

CDH Version: Except 5.15.2, other CDH 5.X, CDH 6.X is not supported

---
### **2. YARN Client Mode on CDH**
-------------------------------------------------------
#### (1) Python Example
##### Step 0: Prepare Environment
- Install Analytics Zoo in the created conda environment via pip:

  ```bash
  pip install analytics-zoo
  ```

  View the [Python User Guide](./python.md) for more details.
  
- Install needed Python dependencies in the created conda environment via pip:

  ```bash
  pip install analytics-zoo[ray] # install either version 0.9 or latest nightly build
  pip install torch==1.7.1 torchvision==0.8.2
  pip install six cloudpickle
  pip install jep==3.9.0
  ```
  
##### **Step 1: Write a Python example script**

- We recommend using `init_orca_context` at the very beginning of your code to initiate and run Analytics Zoo on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn);
- By specifying cluster_mode to be "yarn-client", `init_orca_context` would automatically prepare the runtime Python environment, detect the current Hadoop configurations from `HADOOP_CONF_DIR` and initiate the distributed execution engine on the underlying YARN cluster. View [Orca Context](../Orca/Overview/orca-context.md) for more details.
- Create and write a script with python:

```bash
vim script.py
```
Add the following code in your just created script.py file:
```python
from zoo.orca import init_orca_context
from zoo.orca.learn.pytorch import Estimator 
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch 
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

# init orca context
sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2)

# define the model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
model = LeNet()
model.train()
criterion = nn.NLLLoss()
adam = torch.optim.Adam(model.parameters(), 0.001)

#Define Train Dataset

torch.manual_seed(0)
dir='./'

batch_size=64
test_batch_size=64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dir, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=False)

#create an estimator
est = Estimator.from_torch(model=model, optimizer=adam, loss=criterion, metrics=[Accuracy()])

# fit and evaluate using the Estimator
est.fit(data=train_loader, epochs=10, validation_data=test_loader,
        checkpoint_trigger=EveryEpoch())

result = est.evaluate(data=test_loader)
for r in result:
    print(r, ":", result[r])
    
stop_orca_context()
```
- You may define your model, loss and optimizer in the same way as in any standard (single node) PyTorch program.

- You can define the dataset using standard [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html). Alternatively, we can also use a [Data Creator Function](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/pytorch_lenet_mnist_data_creator_func.ipynb) or [Orca XShards](../Overview/data-parallel-processing) as the input data, especially when the data size is very large)

##### **Step 2: Run your script**

- You can then simply run the Analytics Zoo program in the just written Python script (script.py) in the command line:

  ```bash
  python script.py
  ```
---
### **3. YARN Cluster Mode on CDH**

Follow the steps below if you need to run Analytics Zoo in [YARN cluster mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

- Download and extract [Spark](https://spark.apache.org/downloads.html). You are recommended to use [Spark 2.4.3](https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz). Set the environment variable `SPARK_HOME`:

  ```bash
  export SPARK_HOME=the root directory where you extract the downloaded Spark package
  ```

- Download and extract [Analytics Zoo](../release.md). Make sure the Analytics Zoo package you download is built with the compatible version with your Spark. Set the environment variable `ANALYTICS_ZOO_HOME`:

  ```bash
  export ANALYTICS_ZOO_HOME=the root directory where you extract the downloaded Analytics Zoo package
  ```
- Before running the following two examples, please download [dataset of MNIST](http://yann.lecun.com/exdb/mnist/) on Cloudra Manager and CDH

#### (1) Python Example  
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  PYSPARK_PYTHON=./environment/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --master yarn \
    --deploy-mode cluster \
    --executor-cores 44 \
    --num-executors 3 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
  analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
    -f hdfs://172.16.0.105:8020/mnist \
    -b 132 \
    -e 3
  ```
If run success, you would see the output like:
> final status: SUCCEEDED

and then check the log detail using the following given URL in the output.

#### (2) Scala Example
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  # Spark yarn cluster mode, please make sure the right HADOOP_CONF_DIR is set
  ${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh \
    --master yarn \
    --deploy-mode cluster \
    --executor-cores 44 \
    --num-executors 3 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
    -f hdfs://172.16.0.105:8020/mnist \
    -b 132 \
    -e 3
  ```
If run success, you would see the output like:
> final status: SUCCEEDED

and then check the log detail using the following given URL in the output.
