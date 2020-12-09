#!/bin/bash

set -e

echo "#1 start example for MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]
then
    echo "MNIST already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data
unset ZOO_NUM_MKLTHREADS

now=$(date "+%s")
time1=$((now-start))

echo "#2 start example for orca MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]
then
    echo "MNIST already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/mnist/lenet_mnist.py --dir analytics-zoo-data/data
unset ZOO_NUM_MKLTHREADS

now=$(date "+%s")
time2=$((now-start))

echo "#3 start example for orca Cifar10"
#timer
start=$(date "+%s")
if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/data ]
then
    echo "Cifar10 already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10
    unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.zip
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.py

now=$(date "+%s")
time3=$((now-start))

echo "#4 start example for imagenet"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/imagenet-1k.tar.gz ]
then
    echo "imagenet-1k.tar.gz already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/imagenet-1k.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/imagenet-1k.tar.gz -C analytics-zoo-data/data/
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/imagenet/main.py analytics-zoo-data/data/imagenet-small --batch-size 16 --epochs 1
unset ZOO_NUM_MKLTHREADS

now=$(date "+%s")
time4=$((now-start))

echo "#5 start example for resnet_finetune"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/dogs_cats.zip ]
then
    echo "dogs_cats.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/dogs_cats.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/dogs_cats.zip -d analytics-zoo-data/data/dogs_cats
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/resnet_finetune/resnet_finetune.py analytics-zoo-data/data/dogs_cats
unset ZOO_NUM_MKLTHREADS

now=$(date "+%s")
time5=$((now-start))

echo "#1 MNIST example time used:$time1 seconds"
echo "#2 orca MNIST example time used:$time2 seconds"
echo "#3 orca Cifar10 example time used:$time3 seconds"
echo "#4 imagenet example time used:$time4 seconds"
echo "#5 resnet_finetune example time used:$time5 seconds"
