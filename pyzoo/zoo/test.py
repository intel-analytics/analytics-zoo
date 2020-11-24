#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import tensorflow as tf
import numpy as np
import os

from zoo import init_nncontext
from zoo.orca.learn.tf.estimator import Estimator
import tensorflow_datasets as tfds
sc = init_nncontext()

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

def model_zoo():
    """Autoencoder"""
    input_1 = tf.keras.Input(shape=(784,), name='input_1')
    encoder_1 = tf.keras.layers.Dense(1024, name='encoder_1', activation='relu')(input_1)
    encoder_2 = tf.keras.layers.Dense(256, name='encoder_2', activation='relu')(encoder_1)
    encoder_3 = tf.keras.layers.Dense(128, name='encoder_3', activation='relu')(encoder_2)

    decoder_1 = tf.keras.layers.Dense(128, name='decoder_1', activation='relu')(encoder_3)
    decoder_2 = tf.keras.layers.Dense(256, name='decoder_2', activation='relu')(decoder_1)
    decoder_3 = tf.keras.layers.Dense(1024, name='decoder_3', activation='relu')(decoder_2)
    decoder_4 = tf.keras.layers.Dense(784, name='decoder_4', activation='relu')(decoder_3)

    """DNN 3"""
    dense_3_1 = tf.keras.layers.Dense(128, name='dense_3_1', activation='relu')(encoder_3)
    dense_3_2 = tf.keras.layers.Dense(64, name='dense_3_2', activation='relu')(dense_3_1)
    dense_3_3 = tf.keras.layers.Dense(32, name='dense_3_3', activation='relu')(dense_3_2)
    dense_3_3 = tf.keras.layers.BatchNormalization()(dense_3_3)
    dense_3_3 = tf.keras.layers.Dropout(0.2, seed=1)(dense_3_3)

    output = tf.keras.layers.Dense(10, name='output', activation='softmax')(dense_3_3)

    model = tf.keras.Model(inputs=[input_1],
                           outputs=[decoder_4, output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss={
            'decoder_4': 'mse',
            'output': 'sparse_categorical_crossentropy',
        }
        # metrics=['accuracy']
    )

    return model


def get_data():

    length = 1000
    x = np.random.randn(length, 4)
    y = np.random.randint(0, 1, size=(length, 1))
    return x, y


def get_mnist():
    (train_feature, train_label), _ = tf.keras.datasets.mnist.load_data()

    train_feature = train_feature / 255.0

    x = np.reshape(train_feature, newshape=(-1, 28 * 28))
    y = train_label

    return x, (x, y)



keras_model = model_zoo()

def preprocess(data):
    image = tf.cast(data["image"], tf.float32) / 255.
    image = tf.reshape(image, shape=(784,))
    return image, (image, data['label'])

length = 10
mnist_train = tfds.load(name="mnist", split="train")
mnist_test = tfds.load(name="mnist", split="test")


mnist_train = mnist_train.map(preprocess)
mnist_test = mnist_test.map(preprocess)

keras = True

x = np.random.randn(120, 784).astype(np.float32)
y = np.random.randint(0, 10, (120,1))
# mnist_train = tf.data.Dataset.from_tensor_slices((x, (x, y)))
mnist_train = mnist_train.take(120)

if keras:
    mnist_train = mnist_train.batch(120)
    history = keras_model.fit(mnist_train, epochs=10)
    print(history.history['loss'])
    keras_model.evaluate(x, (x, y))
else:
    print(keras_model.get_weights()[0][0])
    estimator = Estimator.from_keras(keras_model=keras_model)
    estimator.fit(mnist_train, batch_size=120, epochs=10)
    keras_model.evaluate(x, (x, y))


