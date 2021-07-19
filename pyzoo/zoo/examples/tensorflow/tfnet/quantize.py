import tensorflow as tf

from zoo.orca import init_orca_context
from zoo.tfpark import TFNet
import numpy as np

sc = init_orca_context()
import os
os.environ["OMP_NUM_THREADS"] = "4"

(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

test_images = np.reshape(test_images, (-1, 28, 28, 1))

with tf.Graph().as_default():
    input1 = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    flatten = tf.layers.flatten(input1)
    hidden = tf.layers.dense(input1, 128)
    hidden = tf.layers.dense(hidden, 128)
    hidden = tf.layers.dense(hidden, 128)

    output = tf.layers.dense(hidden, 10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tfnet = TFNet.from_session(sess, inputs=[input1], outputs=[output],
                                   quantize=True,
                                   quantize_args=dict(
                                       q_data=(test_images[:100], test_labels[:100]),
                                       e_data=(test_images[:100], test_labels[:100])))
    import time
    start = time.time()
    result = tfnet.predict(test_images)
    preds = result.collect()
    end = time.time()
    print(f"prediction done using {end - start}s")

