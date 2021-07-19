# TFPark API

This is an example to demonstrate how to use Analytics-Zoo's TFPark Scala API to run distributed
Tensorflow Spark/BigDL.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) to install analytics-zoo via __download the prebuilt package__ or __download analytics zoo source__ and __build with script__.

## Prepare MNIST Data
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. /tmp/mnist).

There're four files. **train-images-idx3-ubyte** contains train images,
**train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images
 and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the
 download page.

After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed
to **train-images.idx3-ubyte**. Please change the name back before you run the example.

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and add
the `research/slim` directory to `PYTHONPATH`.

```bash

git clone https://github.com/tensorflow/models/

export PYTHONPATH=$PWD/models/research/slim:$PYTHONPATH
```

## Export lenet Tensorflow model

Export lenet Tensorflow model to local directory for later training:

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package or the dist folder if you build from source
export SPARK_HOME=... # spark home

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
export_lenet.py
```
The `export_lenet.py` script would create inputs of placeholders with corresponding shapes and pass to the Tensorflow model,then export model to a specified folder.(The default folder is "/tmp/lenet_export")

## Run the Tensorflow training on Spark local

Run training with this example script:
```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package or the dist folder if you build from source
export SPARK_HOME=... # spark home

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
 --master local[4] \
 --driver-memory 20g \
 --class com.intel.analytics.zoo.examples.tfpark.Train \
 --model /tmp/lenet_export \
 -d /tmp/mnist \
 -b 280 \
 -e 5
```
__Options:__
* `--model` The exported Tensorflow model directory.
* `-d` Mnist data path.
* `-b` Batch size.
* `-e` Training epoches.