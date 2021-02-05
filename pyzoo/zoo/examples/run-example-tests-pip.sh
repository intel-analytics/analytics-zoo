#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "#1 start example test for textclassification"
start=$(date "+%s")

# Data preparation
if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then
    echo "analytics-zoo-data/data/20news-18828.tar.gz already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/20news-18828.tar.gz -C analytics-zoo-data/data/
fi

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 2 \
    --batch_size 112 \
    --data_path analytics-zoo-data/data/20news-18828 \
    --embedding_path analytics-zoo-data/data/glove.6B
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "textclassification failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))

echo "#2 start example test for image-classification"
#timer
start=$(date "+%s")
echo "check if model directory exists"
if [ ! -d analytics-zoo-models ]
then
    mkdir analytics-zoo-models
fi
if [ -f analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model ]
then
    echo "analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model\
    -P analytics-zoo-models
fi
export SPARK_DRIVER_MEMORY=10g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/imageclassification/predict.py \
    -f ${HDFS_URI}/kaggle/train_100 \
    --model analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model \
    --topN 5
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "imageclassification failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))

echo "#3 start example test for autograd"
#timer
start=$(date "+%s")

export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py 
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "autograd-custom failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/customloss.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "autograd_customloss failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now-start))

echo "#4 start example test for objectdetection"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model ]
then
    echo "analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    -P analytics-zoo-models
fi

export SPARK_DRIVER_MEMORY=10g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/objectdetection/predict.py \
    analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    ${HDFS_URI}/kaggle/train_100 \
    /tmp
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "objectdetection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time4=$((now-start))

echo "#5 start example test for nnframes"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

if [ -f analytics-zoo-data/data/dogs-vs-cats/train.zip ]
then
   echo "analytics-zoo-data/data/dogs-vs-cats/train.zip already exists."
else
   # echo "Downloading dogs and cats images"
   wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip\
    -P analytics-zoo-data/data/dogs-vs-cats
   unzip analytics-zoo-data/data/dogs-vs-cats/train.zip -d analytics-zoo-data/data/dogs-vs-cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.7* analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.7* analytics-zoo-data/data/dogs-vs-cats/samples

   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/dogs
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/demo/cats
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/demo/dogs
   # echo "Finished downloading images"
fi

export SPARK_DRIVER_MEMORY=20g

echo "start example test for nnframes imageInference"
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/ImageInferenceExample.py \
    -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    -f ${HDFS_URI}/kaggle/train_100

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_imageInference failed"
    exit $exit_status
fi

echo "start example test for nnframes finetune"
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
    -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    -f analytics-zoo-data/data/dogs-vs-cats/samples

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_finetune failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
    -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    -f analytics-zoo-data/data/dogs-vs-cats/samples

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_imageTransferLearning failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))

echo "#6 start example test for inceptionv1 training"
#timer
start=$(date "+%s")
export MASTER=local[4]
export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/inception/inception.py \
   --maxIteration 20 \
   -b 8 \
   -f ${HDFS_URI}/imagenet-mini
exit_status=$?
unset MASTER
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "inceptionv1 training failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time6=$((now-start))

echo "#8 start example test for tensorflow"
#timer
start=$(date "+%s")
echo "start example test for tensorflow tfnet"
if [ -f analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz \
    -P analytics-zoo-models
   tar zxf analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -C analytics-zoo-models/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfnet/predict.py \
    --image ${HDFS_URI}/kaggle/train_100 \
    --model analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tfnet failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY

echo "start example test for tensorflow distributed_training"

if [ -d analytics-zoo-models/model ]
then
    echo "analytics-zoo-models/model/research/slim already exists."
else
    git clone https://github.com/tensorflow/models/ analytics-zoo-models
    export PYTHONPATH=$PYTHONPATH:`pwd`/analytics-zoo-models/model/research:`pwd`/analytics-zoo-models/model/research/slim
fi

export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training train_lenet failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training evaluate_lenet failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_dataset.py 5

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark keras keras_dataset failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_ndarray.py 5

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark keras keras_ndarray failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_dataset.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark estimator estimator_dataset failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_inception.py \
    --image-path analytics-zoo-data/data/dogs-vs-cats/demo --num-classes 2

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark estimator estimator_inception failed"
    exit $exit_status
fi

sed "s/MaxIteration(1000)/MaxIteration(5)/g;s/range(20)/range(2)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_and_evaluate.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_tmp.py

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/gan/gan_train_tmp.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark gan gan_train failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time8=$((now-start))

echo "#9 start test for anomalydetection"
#timer
start=$(date "+%s")
# prepare data
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]
then
    echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P analytics-zoo-data/data/NAB/nyc_taxi/
fi
sed "s/model.predict(test)/model.predict(test, batch_per_thread=56)/" ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection.py > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection2.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection2.py \
    --nb_epoch 1 \
    -b 1008 \
    --input_dir analytics-zoo-data//data/NAB/nyc_taxi/nyc_taxi.csv
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomalydetection failed"
    exit $exit_status
fi
now=$(date "+%s")
time9=$((now-start))


echo "#10 start example test for qaranker"
start=$(date "+%s")

if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/WikiQAProcessed.zip ]
then
    echo "analytics-zoo-data/data/WikiQAProcessed.zip already exists"
else
    echo "downloading WikiQAProcessed.zip"
    wget -nv $FTP_URI/analytics-zoo-data/WikiQAProcessed.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/WikiQAProcessed.zip -d analytics-zoo-data/data/
fi

# Run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/qaranker/qa_ranker.py \
    --nb_epoch 2 \
    --batch_size 112 \
    --data_path analytics-zoo-data/data/WikiQAProcessed \
    --embedding_file analytics-zoo-data/data/glove.6B/glove.6B.50d.txt
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "qaranker failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time10=$((now-start))

echo "#12 start example test for vnni/openvino"
start=$(date "+%s")
if [ -d analytics-zoo-models/vnni ]
then
   echo "analytics-zoo-models/resnet_v1_50.xml already exists."
else
   wget $FTP_URI/analytics-zoo-models/openvino/vnni/resnet_v1_50.zip \
    -P analytics-zoo-models
    unzip -q analytics-zoo-models/resnet_v1_50.zip -d analytics-zoo-models/vnni
fi
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data
fi
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/vnni/openvino/predict.py \
    --model analytics-zoo-models/vnni/resnet_v1_50.xml \
    --image analytics-zoo-data/data/object-detection-coco

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "vnni/openvino failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time12=$((now-start))

echo "#13 start example test for streaming Object Detection"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
fi

if [ -f analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model ]
then
    echo "analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
     -P analytics-zoo-models
fi

mkdir -p output
mkdir -p stream
export SPARK_DRIVER_MEMORY=2g
while true
do
   temp1=$(find analytics-zoo-data/data/object-detection-coco -type f|wc -l)
   temp2=$(find ./output -type f|wc -l)
   temp3=$(($temp1+$temp1))
   if [ $temp3 -eq $temp2 ];then
       kill -9 $(ps -ef | grep streaming_object_detection | grep -v grep |awk '{print $2}')
       rm -r output
       rm -r stream
   break
   fi
done  &
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/streaming_object_detection.py \
    --streaming_path ./stream \
    --model analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
    --output_path ./output  &
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/image_path_writer.py \
    --streaming_path ./stream \
    --img_path analytics-zoo-data/data/object-detection-coco

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "streaming Object Detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time13=$((now-start))

echo "#14 start example test for streaming Text Classification"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/textclassification/streaming_text_classification.py \
    --model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
    --index_path analytics-zoo-data/data/streaming/text-model/word_index.txt \
    --input_file analytics-zoo-data/data/streaming/text-model/textfile/ > 1.log &
while :
do
echo "I am strong and I am smart" >> analytics-zoo-data/data/streaming/text-model/textfile/s
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep streaming_text_classification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "streaming Text Classification failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time14=$((now-start))

echo "#15 start test for orca data"
#timer
start=$(date "+%s")
# prepare data
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]
then
    echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P analytics-zoo-data/data/NAB/nyc_taxi/
fi

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/data/spark_pandas.py \
    -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca data failed"
    exit $exit_status
fi
now=$(date "+%s")
time15=$((now-start))

echo "#16 start test for orca tf imagesegmentation"
#timer
start=$(date "+%s")
# prepare data
if [ -f analytics-zoo-data/data/carvana ]
then
    echo "analytics-zoo-data/data/carvana already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/carvana/train.zip \
    -P analytics-zoo-data/data/carvana/
    wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.zip \
    -P analytics-zoo-data/data/carvana/
    wget $FTP_URI/analytics-zoo-data/data/carvana/train_masks.csv.zip \
    -P analytics-zoo-data/data/carvana/
fi

# Run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/image_segmentation/image_segmentation.py \
    --file_path analytics-zoo-data/data/carvana --epochs 1 --non_interactive
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca tf imagesegmentation failed"
    exit $exit_status
fi
now=$(date "+%s")
time16=$((now-start))

echo "#17 start test for orca tf transfer_learning"
#timer
start=$(date "+%s")
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/transfer_learning/transfer_learning.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca tf transfer_learning failed"
    exit $exit_status
fi
now=$(date "+%s")
time17=$((now-start))

echo "#18 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
sed "s/epochs=100/epochs=10/g" \
  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  >${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/tmp.py
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca tf basic_text_classification failed"
    exit $exit_status
fi
now=$(date "+%s")
time18=$((now-start))

echo "#19 start test for orca bigdl attention"
#timer
start=$(date "+%s")
#run the example
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 200/g;s/max_len = 200/max_len = 20/g;s/hidden_size=128/hidden_size=8/g;s/memory=\"100g\"/memory=\"20g\"/g;s/driver_memory=\"20g\"/driver_memory=\"3g\"/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/transformer.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/tmp.py
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/tmp.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca bigdl attention failed"
    exit $exit_status
fi
now=$(date "+%s")
time19=$((now-start))

echo "#20 start example test for autograd"
#timer
start=$(date "+%s")
echo "#20.1 start example test for orca custom layer"
python  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/autograd/custom.py --cluster_mode local --nb_epoch 10

echo "#20.2 start example test for orca customloss"
python  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/autograd/customloss.py --cluster_mode local --nb_epoch 10

now=$(date "+%s")
time20=$((now - start))

clear_up

echo "#1 textclassification time used: $time1 seconds"
echo "#2 imageclassification time used: $time2 seconds"
echo "#3 autograd time used: $time3 seconds"
echo "#4 objectdetection time used: $time4 seconds"
echo "#5 nnframes time used: $time5 seconds"
echo "#6 inceptionV1 training time used: $time6 seconds"
#echo "#7 pytorch time used: $time7 seconds"
echo "#8 tensorflow time used: $time8 seconds"
echo "#9 anomalydetection time used: $time9 seconds"
echo "#10 qaranker time used: $time10 seconds"
echo "#12 vnni/openvino time used: $time12 seconds"
echo "#13 streaming Object Detection time used: $time13 seconds"
echo "#14 streaming text classification time used: $time14 seconds"
echo "#15 orca data time used:$time15 seconds"
echo "#16 orca tf imagesegmentation time used:$time16 seconds"
echo "#17 orca tf transfer_learning time used:$time17 seconds"
echo "#18 orca tf basic_text_classification time used:$time18 seconds"
echo "#19 orca bigdl attention time used:$time19 seconds"
echo "#20 orca tf autograd time used:$time20 seconds"
