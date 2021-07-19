## Overview
This is a Scala example for image inference with Spark Structured Streaming.

Analytics Zoo provides the DataFrame-based API for image reading, preprocessing, model training
and inference. The related classes followed the typical estimator/transformer pattern of Spark
ML and can be used in a standard Spark ML pipeline.

This example contains two programs:
1. ImagePathWriter is used as the producer, it will write the image path to a specific folder
 every a few seconds to simulate the streaming input.
2. ImageStructedStreaming will load the pretrained model and run inference in Spark Structured
 Streaming.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Run the example

1. Prepare pre-trained model and defenition file.
You can download the pre-trained model from
[Analytics Zoo](https://analytics-zoo.github.io/0.4.0/#ProgrammingGuide/image-classification/).
Models like Inception or ResNet can work with the preprocessing steps defined in ImageStructedStreaming.

2. Prepare dataset
You can use your own image data (JPG or PNG), or some images from imagenet-2012 validation
dataset <http://image-net.org/download-images> to run the example. 

3. Run this example

Run the following command for Spark local mode, adjust the memory size according to your image:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[1] \
    --driver-memory 5g \
    --class com.intel.analytics.zoo.examples.nnframes.stucturedStreaming.ImageStructuredStreamingExample \
    --streamingPath /home/yuhao/workspace/github/hhbyyh/Test/ZooExample/src/streaming/test/folder \
    --model /home/yuhao/workspace/model/analytics-zoo_resnet-50_imagenet_0.1.0.model
```

Then run the following command to start the producer

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[1] \
    --driver-memory 1g \
    --class com.intel.analytics.zoo.examples.nnframes.stucturedStreaming.ImagePathWriter \
    --imageSourcePath /home/yuhao/workspace/data/imagenet/predict_m \
    --streamingPath /home/yuhao/workspace/github/hhbyyh/Test/ZooExample/src/streaming/test/folder
```
 

From the inference, you should see something like this in the console:

```
...

Batch: 3
-------------------------------------------
+--------------------------------------------------------------------------+------+
|value                                                                     |output|
+--------------------------------------------------------------------------+------+
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000231.JPEG|125   |
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000041.JPEG|431   |
+--------------------------------------------------------------------------+------+

-------------------------------------------
Batch: 4
-------------------------------------------
+--------------------------------------------------------------------------+------+
|value                                                                     |output|
+--------------------------------------------------------------------------+------+
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000363.JPEG|133   |
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000198.JPEG|16    |
+--------------------------------------------------------------------------+------+

-------------------------------------------
Batch: 5
-------------------------------------------
+--------------------------------------------------------------------------+------+
|value                                                                     |output|
+--------------------------------------------------------------------------+------+
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000154.JPEG|115   |
|/home/yuhao/workspace/data/imagenet/predict_m/ILSVRC2012_val_00000477.JPEG|584   |
+--------------------------------------------------------------------------+------+

...

```
