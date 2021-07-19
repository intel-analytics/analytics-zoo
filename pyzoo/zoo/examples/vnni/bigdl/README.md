# Summary
We hereby illustrate the support of [VNNI](https://en.wikichip.org/wiki/x86/avx512vnni) using BigDL [MKL-DNN](https://github.com/intel/mkl-dnn) as backend in Analytics Zoo, which aims at accelerating inference by utilizing low numerical precision (Int8) computing.

Int8 quantized models can generally give you better performance on Intel Xeon scalable processors.

## Download Analytics Zoo and the pre-trained model
- You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.
- Download pre-trained int8 quantized ResNet50 model from [here](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/image-classification/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model).

## Examples
This folder contains three examples for BigDL VNNI support:
- [Predict](#predict)
- [Perf](#perf)

__Remarks:__
- If you are using an int8 quantized model pre-trained in Analytics Zoo, you can find the following info in console log:
```
INFO  ImageModel$:132 - Loading an int8 convertible model. Quantize to an int8 model for better performance
```
- You may need to enlarge memory configurations depending on the size of your input data.
- You can set `-Dbigdl.mklNumThreads` if necessary.

---
### Predict
This example demonstrates how to do image classification with the pre-trained int8 model.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export SPARK_DRIVER_MEMORY=2g

image_path=the folder path containing images
model_path=the path to the downloaded int8 model

python Predict.py --model ${model_path} --image ${image_path}

```

__Options:__
- `--folder`: The local folder path that contains images for prediction.
- `--model`: The path to the downloaded int8 model.
- `--topN`: The top N classes with highest probabilities as output. Default is 5.

__Sample console log output__:
```
Image : file:/home/aqtjin/val_bmp_32/ILSVRC2012_val_00000027.bmp, top 5 prediction result
        white wolf, Arctic wolf, Canis lupus tundrarum, 0.994037
        timber wolf, grey wolf, gray wolf, Canis lupus, 0.005738
        Arctic fox, white fox, Alopex lagopus, 0.000108
        Eskimo dog, husky, 0.000046
        coyote, prairie wolf, brush wolf, Canis latrans, 0.000030
Image : file:/home/aqtjin/val_bmp_32/ILSVRC2012_val_00000017.bmp, top 5 prediction result
        starfish, sea star, 0.506320
        sea anemone, anemone, 0.216440
        axolotl, mud puppy, Ambystoma mexicanum, 0.171718
        sea urchin, 0.045972
        goldfish, Carassius auratus, 0.028131
Image : file:/home/aqtjin/val_bmp_32/ILSVRC2012_val_00000014.bmp, top 5 prediction result
        recreational vehicle, RV, R.V., 0.999174
        moving van, 0.000454
        mobile home, manufactured home, 0.000271
        minibus, 0.000049
        ambulance, 0.000022
```

---
### Perf
This example runs in local mode and calculates performance data (i.e. throughput and latency) for the pre-trained int8 model using dummy input.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export SPARK_DRIVER_MEMORY=16g
model_path=the path to the downloaded int8 model

python Perf.py --model ${model_path}
```

__Options:__
- `--model`: The path to the downloaded int8 model.
- `--batch_size`: The batch size of input data. Default is 32.
- `--iteration`: The number of iterations to run the performance test. Default is 1000. The result should be the average of each iteration time cost.

__Sample console log output__:
```
......
Iteration:72, batch 32, takes XXX ms, throughput is XXX imgs/sec
......
Iteration:20, latency for a single image is XXX ms
......
```
