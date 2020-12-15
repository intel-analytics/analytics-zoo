# Image Segmentation with Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed image segmentation training and inference task.

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -n zoo python=3.7
conda activate zoo
pip install tensorflow==1.15
pip install Pillow
pip install pandas
pip install matplotlib
pip install sklearn
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl
```

Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Data Preparation
You should manually download the dataset from kaggle [carvana-image-masking-challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) and save it to `~/tensorflow_datasets/downloads/manual/carvana`. We will need two files, train.zip and train_masks.zip

## Run example on local
```bash
python image_segmentation.py --cluster_mode local 
```

## Run example on yarn cluster

If have not run image_segmention.py locally, you can run the following command to prepare tf_records file locally and put it to hdfs.
Although tensorflow_datasets support directly perparing data in hdfs, that might generate too much small files that will harm your
hdfs performance.

```bash
python carvana_datasets.py
hadoop fs -put ~/tensorflow_datasets/ /tensorflow_datasets/
```


```bash
source ${HADOOP_HOME}/libexec/hadoop-config.sh # setting HADOOP_HDFS_HOME, LD_LIBRARY_PATH, etc
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server

CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python image_segmentation.py --cluster_mode yarn 
```

Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`.
* `--file_path` The path to carvana train.zip, train_mask.zip and train_mask.csv.zip. Default to be `/tmp/carvana/`.
* `--epochs` The number of epochs to train the model. Default to be 8.
* `--batch_size` Batch size for training and prediction. Default to be 8.
