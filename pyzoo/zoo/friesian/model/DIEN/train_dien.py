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
import time

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from zoo.orca.data.file import exists, makedirs
from zoo.orca.learn.tf.estimator import Estimator
from train_utils import *
from zoo.orca import init_orca_context, stop_orca_context

if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Tensorflow DIEN Training/Inference')
    # parameters
    parser.add_argument('--cluster_mode', type=str, default="spark-submit",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                      help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--model_type', default='DIEN', type=str,
                        help='model type: DIEN-gru-att-augru (default)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    spark = SparkSession(sparkContext=sc)

    train_data, test_data, n_uid, n_mid, n_cat = load_dien_data(spark, args.data_dir)

    model = build_model(args.model_type, n_uid, n_mid, n_cat, args.lr)
    [inputs, feature_cols] = align_input_features(model)

    estimator = Estimator.from_graph(inputs=inputs, outputs=[model.y_hat],
                                     labels=[model.target_ph], loss=model.loss,
                                     optimizer=model.optim, model_dir=args.model_dir,
                                     metrics={'loss': model.loss, 'accuracy': model.accuracy})

    for i in range(args.epochs):
        estimator.fit(train_data, epochs=1, batch_size=args.batch_size, feature_cols=feature_cols,
                  label_cols=['label'], validation_data=test_data)

        result = estimator.evaluate(test_data, args.batch_size, feature_cols=feature_cols,
                            label_cols=['label'])
        print('test result:', result)
        prediction_df = estimator.predict(test_data, feature_cols=feature_cols)
        transform_label = udf(lambda x: int(x[1]), "int")
        prediction_df = prediction_df.withColumn('label_t', transform_label(col('label')))
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                          labelCol="label_t",
                                          metricName="areaUnderROC")
        auc = evaluator.evaluate(prediction_df)
        print("test AUC score is: ", auc)

    cpkts_dir = os.path.join(args.model_dir, 'cpkts/')
    if not exists(cpkts_dir): makedirs(cpkts_dir)
    snapshot_path = cpkts_dir + "cpkt_noshuffle_" + args.model_type
    estimator.save_tf_checkpoint(snapshot_path)
    time_end = time.time()
    print('perf { total time: %f }' % (time_end - time_start))

    stop_orca_context()
