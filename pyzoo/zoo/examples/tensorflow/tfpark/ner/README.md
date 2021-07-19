## NER distributed inference example
This example demonstrates how to use Analytics Zoo tfpark API to do distributed NER inference using a pre-trained model.

---
## Download Analytics Zoo nightly-build package
You can download our recent nightly-build packages prebuilt with different Spark versions from [here](https://analytics-zoo.github.io/master/#release-download/#release-050-nightly-build).

Please choose the one ends with `dist.zip` with recent timestamps (e.g. 20190313), such as [this](https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/zoo/analytics-zoo-bigdl_0.7.2-spark_2.3.1/0.5.0-SNAPSHOT/analytics-zoo-bigdl_0.7.2-spark_2.3.1-0.5.0-20190313.181402-41-dist.zip) for Spark 2.3.

---
## Download the pre-trained model
- Download the pre-trained model from [here](http://nervana-modelzoo.s3.amazonaws.com/NLP/ner/model.h5).
- Download the model topology info from [here](http://nervana-modelzoo.s3.amazonaws.com/NLP/ner/model_info.dat).

The model is pre-trained on four entities: LOC, PER, ORG and MISC.

---
## Environment Preparation
You need to install `h5py` and `tensorflow==1.10.0` on the __driver node__.
```bash
pip install h5py
pip install tensorflow==1.10.0
```

---
## Running command
Set the following environment variables:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the parent folder where you extract the downloaded Analytics Zoo zip package
```
Not that we only support Python 3.5 and Python, so you need to make sure that pyspark runs on the proper Python version.
```bash
export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
```
You may need to modify the python3 path according to your driver and worker node above.
 
Run the following command for Spark cluster:
```bash
MASTER=...

${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --driver-cores 8 \
    --executor-memory 2g \
    --executor-cores 8 \
    --total-executor-cores 24 \
    ner_inference.py \
    --model_path model.h5 \
    --model_info_path model_info.dat \
    --input_path ...
```

__Options:__
* `--model_path` and `model_info_path`: The path to `model.h5` and `model_info.dat` downloaded from [here](#download-the-pre-trained-model).
* `--seq_length`: The maximum sequence length of inputs. Default is 30.
* `--input_path`: The path to the input txt file if any. Each line should be an input sentence for NER. It can be an HDFS path. If not specified, the program will perform NER on two example sentences and print the result.


## Results
Given RDD of string as input, the result will be RDD of list of entity tags.

We print out several sample outputs to console:
```
John	B-PER	
is	O	
planning	O	
a	O	
visit	O	
to	O	
London	B-LOC	
on	O	
October	O	

even	O	
though	O	
Intel	B-ORG	
is	O	
a	O	
big	O	
organization	O	
purchasing	O	
Mobileye	B-ORG	
last	O	
year	O	
had	O	
a	O	
huge	O	
positive	O	
impact	O	
```