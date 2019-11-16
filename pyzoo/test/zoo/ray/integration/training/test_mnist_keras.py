import ray
import tensorflow as tf
import numpy as np


from zoo.ray.data.dataset import RayDataSet
from zoo.ray.distribute.training import RayModel

images = tf.keras.layers.Input((28, 28, 1))
target = tf.keras.layers.Input((1, ))
input1 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(images)
tf.keras.layers.BatchNormalization(axis=-1)(input1)
dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)
dropout = tf.keras.layers.Dropout(0.2)(dense)
dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)

dense2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0, 1)),
                      name='transpose')(dense2)
keras_model = tf.keras.Model(inputs=[images], outputs=[dense2])
keras_model.summary()

keras_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
keras_model.save("/opt/work/tt.model")

num_worker = 2
resource={"trainer": num_worker, "ps": num_worker }
ray.init(local_mode=True, log_to_driver=True, resources=resource)
batch_size = 1280

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train) = np.random.uniform(0, 1, size=(600, 28, 28, 1)), np.random.uniform(0, 1, size=(600, 1))
x_train = x_train.reshape((-1, 28, 28, 1))
y_train = y_train.reshape((-1, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_test = y_test.reshape((-1, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0


rayModel = RayModel.from_keras_model(keras_model)

rayModel.fit(x=x_train,
             y=y_train,
             num_worker=num_worker,
             batch_size=128,
             steps=400,
             strategy="ps")

# slow if the batch is small
print("ACC: {}".format(rayModel.evaluate(x=x_test, y=y_test, batch_size=1000)))
# ACC: 0.9697999954223633

ray.shutdown()
