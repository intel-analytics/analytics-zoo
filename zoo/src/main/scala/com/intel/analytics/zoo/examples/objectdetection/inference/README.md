## Object Detection example
This example illustrates how to detect objects in image with pre-trained model

### Run steps
1. Prepare pre-trained models

Download pre-trained models from [Object Detection](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/object-detection.md)

2. Prepare predict dataset

Put your image data for prediction in the ./image folder.

3. Run the example

```bash
master=... // spark master

modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported

imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster.

outputPath=... // output path. Currently only support local file system.

spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.objectdetection.inference.Predict \
${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar --image $imagePath --output $outputPath --modelPath $modelPath --partition 4
```

## Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
