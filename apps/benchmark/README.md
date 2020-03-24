# Analytics-Zoo Benchmark

This is a simple project to measure the performance of Analytics-Zoo with BigDL, TFNet, TorchNet and OpenVINO backend. Currently, it supports image classification inference.

The run.sh gives an example of how to use this project. You can modify the batch size, iteration, model path and quantize or not.

## Model Zoo

Model Zoo & Links

1. [Analytics-Zoo & BigDL](https://analytics-zoo.github.io/master/#ProgrammingGuide/image-classification/#download-link)
2. [TensorFlow](https://github.com/tensorflow/models/tree/master/research/slim)
3. [PyTorch](https://pytorch.org/docs/stable/torchvision/models.html)
4. [OpenVINO](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)

## How to build

Build jar with following command:

`mvn clean package`

## Usage for Pure JAVA Benchmark

```shell
usage:
       1. type, tf, torch, bigdl, bigdlblas and ov, e,g., bigdl
       2. Model path, e.g., *.model, *.xml
       3. Iteration, e.g., 100
       4. Batch Size, e.g., 32
       as parameters in order. More concretely, you can run this command:
       bash run.sh \\
            bigdl \\
            /path/model \\
            100 \\
            32"
```

`bash run.sh bigdl analytics-zoo_resnet-50_imagenet_0.1.0.model 10 64`

## Usage for Spark OpenVINO Benchmark

Create random data in multiple partitions, then running benchmark with multiple executors.

### Multiple nodes

Zoo will broadcast model to multiple nodes. No further configuration is necessary.

### Single Node

Not recommended. OMP instance may conflict with each other.

_Pls place worker/executors in diferent containers or binding them to different numa node with numactl. Otherwise, OMP related environment may conflict and lead to low CPU usage._

1. Single worker (single slave)
2. Multiple workers (multiple slave instance)

**Note that local model doesn't support multiple executors. Please launch a standalone Spark.**

If your model is too large to Spark broadcast, pls change `spark.rpc.message.maxSize`.

```shell
usage:
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
```

## Other Details

1. This is not the whole pipeline performance script. It only focus how much
   time model computing takes, which means it measures the forward without
   pre-processing. The input of model is a dummy random tensor.

2. Pay attention to `${CPU}` in `run.sh`. Sometimes it will get wrong number because of HT.

3. Change `HYPER=2` in `run.sh` if hyper-threading is on.

4. `warm up` in benchmark. The performance of first several batches may not stable. So, we added 10 batches for warm up. Their performance will not be taken into account.
