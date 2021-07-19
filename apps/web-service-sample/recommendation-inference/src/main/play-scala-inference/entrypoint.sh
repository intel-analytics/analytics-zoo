#!/bin/sh
CONTEXT=$1
CONFIG_ENV=$2
exec java $minHeap $maxHeap -jar /target/scala-2.11/recrnnservice-assembly-1.0-SNAPSHOT.jar