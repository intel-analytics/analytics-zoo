FROM zoo-base:0.2.0-py2

RUN pip2 install --upgrade setuptools && \
    pip2 install numpy six scipy pandas scikit-learn matplotlib jupyter requests && \
    pip2 install flask gevent gunicorn && \
    pip2 install keras==1.2.2 h5py && \
    pip2 install pyspark==2.2.1 bigdl==0.6.0 && \
    pip2 install sagemaker_pyspark==1.1.2 sagemaker-containers==2.0.2 && \
    python2 -m pip install ipykernel && \
    python2 -m ipykernel install --user && \
    (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
    rm -rf /root/.cache

ARG framework_support_installable=analytics-zoo-0.2.0.tar.gz

WORKDIR /root

COPY $framework_support_installable .

RUN framework_support_installable_local=$(basename $framework_support_installable) && \
    \
    pip2 install $framework_support_installable_local && \
    \
    rm $framework_support_installable_local

ENV SAGEMAKER_TRAINING_MODULE zoo.sagemaker.container.training:main
ENV SAGEMAKER_SERVING_MODULE zoo.sagemaker.container.serving:main
