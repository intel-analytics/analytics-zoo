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

import numpy
import os
import random
from model_dien import *
from pyspark.sql.functions import desc, rank, col
from pyspark.sql.window import Window
import sys

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
SEED = 3

def build_model(model_type, n_uid, n_mid, n_cat, lr):
    if model_type == 'DNN':
        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'PNN':
        model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'Wide':
        model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'DIN':
        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                            ATTENTION_SIZE, lr)
    elif model_type == 'DIEN':
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                ATTENTION_SIZE, lr)
    else:
        print("Invalid model_type: %s", model_type)
        sys.exit(1)
    return model

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    tf.set_random_seed(seed)

def align_input_features(model):
    input_phs = [model.uid_batch_ph, model.mid_his_batch_ph, model.cat_his_batch_ph, model.mask,
                 model.seq_len_ph, model.mid_batch_ph, model.cat_batch_ph]
    feature_cols = ['user', 'item_hist_seq', 'category_hist_seq', 'item_hist_seq_mask',
                    'item_hist_seq_len', 'item', 'category']
    if model.use_negsampling:
        input_phs.extend([model.noclk_mid_batch_ph, model.noclk_cat_batch_ph])
        feature_cols.extend(['neg_item_hist_seq', 'neg_category_hist_seq'])
    return [input_phs, feature_cols]

def load_dien_data(spark, data_dir):
    df = spark.read.parquet(data_dir + "/data")
    windowSpec1  = Window.partitionBy("user").orderBy(desc("time"))
    windowSpec2  = Window.partitionBy("user").orderBy("time")
    df = df.withColumn("rank1", rank().over(windowSpec1))
    df = df.withColumn("rank2", rank().over(windowSpec2))
    test_data = df.filter(col('rank1') == 1)
    #test_data = df.filter((col('rank1') == 1) & (col('rank2') > 1))
    train_data = df.subtract(test_data)
    # train_data, test_data = df.randomSplit([0.8, 0.2])
    train_data.write.parquet(data_dir + "/train", mode='overwrite')
    test_data.write.parquet(data_dir + "/test", mode='overwrite')
    train_data = spark.read.parquet(data_dir + "/train")
    test_data = spark.read.parquet(data_dir + "/test")
    userdf = spark.read.parquet(data_dir + "/user_index/*")
    itemdf = spark.read.parquet(data_dir + "/item_index/*")
    catdf = spark.read.parquet(data_dir + "/category_index/*")
    n_uid = userdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    n_mid = itemdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    n_cat = catdf.select("id").agg({"id": "max"}).collect()[0][0] + 1
    return train_data, test_data, n_uid, n_mid, n_cat

