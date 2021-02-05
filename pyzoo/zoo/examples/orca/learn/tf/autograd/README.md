# Transfer Learning with Orca TF Estimator

This is an example to illustrate how to define a custom loss function and ```Lambda``` layer and set tensorboard in Analytics-Zoo's Orca TF Estimator API. See original example [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/autograd).

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -n zoo python=3.7
conda activate zoo
pip install tensorflow==1.15
pip install --pre analytics-zoo
```
Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Run example on local
```bash
python custom.py --cluster_mode local --nb_epoch 5
python customloss.py --cluster_mode local --nb_epoch 5
```

## Run example on yarn cluster
```bash
python custom.py --cluster_mode yarn --nb_epoch 5
python customloss.py --cluster_mode yarn --nb_epoch 5
```

Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`.
* `--nb_epoch` The number of epochs to train for. Default is 500.

## Results
You can find the result for training as follows:
```
2021-02-03 09:23:24 INFO  DistriOptimizer$:427 - [Epoch 5 992/1000][Iteration 159][Wall Clock 4.906765237s] Trained 32.0 records in 0.026909288 seconds. Throughput is 1189.1804 records/second. Loss is 1.1435537. 
2021-02-03 09:23:24 INFO  DistriOptimizer$:427 - [Epoch 5 1024/1000][Iteration 160][Wall Clock 4.932384347s] Trained 32.0 records in 0.02561911 seconds. Throughput is 1249.0676 records/second. Loss is 1.0706859. 
2021-02-03 09:23:24 INFO  DistriOptimizer$:472 - [Epoch 5 1024/1000][Iteration 160][Wall Clock 4.932384347s] Epoch finished. Wall clock time is 4949.199824 ms
```
You can find the result for predict as follows:
```
[[1.9931875]
 [2.110691 ]
 [2.2392426]
 [2.0116081]
 [2.0122721]
         ...]
```
At last, you can find tensorboard log directory at './log'
