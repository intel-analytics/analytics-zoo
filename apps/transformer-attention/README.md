# Transformer-Attention is all you need
Transformer is a neural network architect proposed in the paper "Attention Is All You Need"
(https://arxiv.org/abs/1706.03762). The network architecture is based solely on attention
mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine
translation tasks show the model to be superior in quality while being more parallelizable
and requiring significantly less time to train.

In Multi-head Attention Sentiment Analyisi notebook, we use keras API build the core layer: 
Multi-head Attention layer and apply it to a sentiment analysis example.


## Run with Jupyter
* Analytics-Zoo supports 1) pip install and 2) run with pySpark without pip. Setup would be easy
 following the doc: https://analytics-zoo.github.io/master/#PythonUserGuide/install/, and here's 
 info about how to start jupyter notebook after setup:
 https://analytics-zoo.github.io/master/#PythonUserGuide/run/

* Open the notebook in Jupyter and start your exploration.

```
