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

import argparse

from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.optimizers import *
from zoo.pipeline.api.autograd import *
from zoo.common.nncontext import init_spark_conf
from zoo.common.nncontext import init_nncontext
from zoo.examples.attention.input import *

conf = init_spark_conf()
conf.set("spark.executor.extraJavaOptions", "-Xss512m")
conf.set("spark.driver.extraJavaOptions", "-Xss512m")
sc = init_nncontext(conf)

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('vocab_path', help="Path where the vocab.txt is stored")
parser.add_argument('data_dir', help="Path to store the training data")

args = parser.parse_args()
max_seq_length = 128
num_labels = 2

print('Loading data...')
processor = MrpcProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(args.vocab_path)

def build_sample(feature):
    sample = Sample.from_ndarray([feature.input_ids, feature.segment_ids, feature.pos_ids, feature.input_mask],
                                     feature.label_id)
    return sample


train_examples = processor.get_train_examples(args.data_dir)
train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
train_samples = [build_sample(feature) for feature in train_features]

eval_examples = processor.get_dev_examples(args.data_dir)
eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
eval_samples = [build_sample(feature) for feature in eval_features]

train_data = sc.parallelize(train_samples)
eval_data = sc.parallelize(eval_samples)

token_shape = (max_seq_length,)
position_shape = (max_seq_length,)
segment_shape = (max_seq_length,)
mask_shape = (1, 1, max_seq_length)

token_input = Input(shape=token_shape)
position_input = Input(shape=position_shape)
segment_input = Input(shape=segment_shape)
mask_input = Input(shape=mask_shape)
bert = BERT.init_from_existing_model(args.model_path, input_seq_len=max_seq_length)
O_seq = bert([token_input, segment_input, position_input, mask_input])
O_seq = SelectTable(12)(O_seq)
O_seq = Dropout(0.2)(O_seq)
outputs = Dense(num_labels, "normal", (0.0, 0.02), activation='softmax')(O_seq)

model = Model([token_input, segment_input, position_input, mask_input], outputs)
model.summary()

model.compile(
    optimizer=AdamWeightDecay(lr=2e-5, warmup_portion=0.1, total=343),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.set_gradient_clipping_by_l2_norm(1.0)
batch_size = 32
print('Train...')
model.fit(train_data,
          batch_size=batch_size,
          nb_epoch=3)
print("Train finished.")

print('Evaluating...')
score = model.evaluate(eval_data, batch_size=32)
print("eval_loss is: ", score[0])
print("eval_accuracy is: ", score[1])
