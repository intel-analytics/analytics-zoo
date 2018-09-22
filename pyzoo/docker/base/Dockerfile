FROM ubuntu:14.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

# Set up the program in the image
COPY logistic_regression /opt/work
WORKDIR /opt/work

# analytics zoo env
ARG ANALYTICS_ZOO_VERSION=0.2.0-SNAPSHOT
ARG SPARK_VERSION=2.2.0
ENV ANALYTICS_ZOO_VERSION_ENV   ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION_ENV           ${SPARK_VERSION}
#ENV SPARK_HOME                  /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME          /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                   /opt/jdk
# PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH                        ${JAVA_HOME}/bin:/opt/work:${PATH}

# basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim curl nano wget unzip maven git openssh-server pkg-config \
        unzip zip iputils-ping nginx ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# java
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk
# python
RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties python-pkg-resources && \
    add-apt-repository -y ppa:jonathonf/python-2.7 && \
    apt-get update && \
    apt-get install -y build-essential python python-setuptools python-dev python3-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

#spark
RUN wget https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz

# SSH. Partially taken from https://docs.docker.com/engine/examples/running_ssh_service/
RUN mkdir /var/run/sshd
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Create SSH key.
RUN mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

