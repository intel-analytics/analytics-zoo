#!/bin/bash

imageFolder=$1
output=$2

export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

java -cp ${ANALYTICS_ZOO_JAR} \
         com.intel.analytics.zoo.models.image.objectdetection.common.dataset.RoiImageSeqGenerator \
     -f $imageFolder -o $output