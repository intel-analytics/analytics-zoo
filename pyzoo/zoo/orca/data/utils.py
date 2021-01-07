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
import os
import numpy as np

from zoo.common import get_file_list
import pyarrow as pa

from zoo.util.utils import get_node_ip


def list_s3_file(file_path, env):
    path_parts = file_path.split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = env["AWS_ACCESS_KEY_ID"]
    secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    # file
    if os.path.splitext(file_path)[1] != '':
        return ["s3://" + file_path]
    else:
        keys = []
        resp = s3_client.list_objects_v2(Bucket=bucket,
                                         Prefix=key)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        file_paths = [os.path.join("s3://" + bucket, file) for file in keys]
        return file_paths


def extract_one_path(file_path, env):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    if prefix == "s3":
        file_paths = list_s3_file(file_url_splits[1], env)
    elif prefix == "hdfs":
        import pyarrow as pa
        fs = pa.hdfs.connect()
        if fs.isfile(file_path):
            file_paths = [file_path]
        else:
            file_paths = get_file_list(file_path)
    else:  # Local file path; could be a relative path.
        from os.path import isfile, abspath, join
        if isfile(file_path):
            file_paths = [abspath(file_path)]
        else:
            # An error would be already raised here if the path is invalid.
            file_paths = [abspath(join(file_path, file)) for file in os.listdir(file_path)]
    return file_paths


def check_type_and_convert(data, allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def check_and_convert(convert_data):
        if isinstance(convert_data, np.ndarray):
            return [convert_data]
        elif isinstance(convert_data, tuple) and \
                all([isinstance(d, np.ndarray) for d in convert_data]):
            return _convert_list_tuple(convert_data, allow_tuple=allow_tuple,
                                       allow_list=allow_list)
        elif isinstance(convert_data, list) and \
                all([isinstance(d, np.ndarray) for d in convert_data]):
            return _convert_list_tuple(convert_data, allow_tuple=allow_tuple,
                                       allow_list=allow_list)
        else:
            raise ValueError("value of x and y should be a ndarray, "
                             "a tuple of ndarrays or a list of ndarrays")

    result = {}
    assert isinstance(data, dict), "each shard should be an dict"
    assert "x" in data, "key x should in each shard"
    x = data["x"]
    result["x"] = check_and_convert(x)
    if "y" in data:
        y = data["y"]
        result["y"] = check_and_convert(y)
    return result


def get_spec(allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def _get_spec(data):
        data = check_type_and_convert(data, allow_tuple, allow_list)
        feature_spec = [(feat.dtype, feat.shape[1:])
                        for feat in data["x"]]
        if "y" in data:
            label_spec = [(label.dtype, label.shape[1:])
                          for label in data["y"]]
        else:
            label_spec = None
        return feature_spec, label_spec
    return _get_spec


# todo this might be very slow
def flatten_xy(allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def _flatten_xy(data):
        data = check_type_and_convert(data, allow_tuple, allow_list)
        features = data["x"]

        has_label = "y" in data
        labels = data["y"] if has_label else None
        length = features[0].shape[0]

        for i in range(length):
            fs = [feat[i] for feat in features]
            if has_label:
                ls = [l[i] for l in labels]
                yield (fs, ls)
            else:
                yield (fs,)
    return _flatten_xy


def combine(data_list):
    item = data_list[0]
    if isinstance(item, dict):
        res = {}
        for k, v in item.items():
            res[k] = np.concatenate([data[k] for data in data_list], axis=0)
    elif isinstance(item, list) or isinstance(item, tuple):
        res = []
        for i in range(len(data_list[0])):
            res.append(np.concatenate([data[i] for data in data_list], axis=0))
        if isinstance(item, tuple):
            res = tuple(res)
    elif isinstance(data_list[0], np.ndarray):
        res = np.concatenate(data_list, axis=0)
    else:
        raise ValueError(
            "value of x and y should be an ndarray, a dict of ndarrays, a tuple of ndarrays"
            " or a list of ndarrays, please check your input")
    return res


def ray_partition_get_data_label(partition_data,
                                 allow_tuple=True,
                                 allow_list=True,
                                 has_label=True):
    """
    :param partition_data: The data partition from Spark RDD, which should be a list of records.
    :param allow_tuple: Boolean. Whether the model accepts a tuple as input. Default is True.
    :param allow_list: Boolean. Whether the model accepts a list as input. Default is True.
    :param has_label: Boolean. Whether the data partition contains labels.
    :return: Concatenated data for the data partition.
    """
    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]

    data = _convert_list_tuple(combine(data_list),
                               allow_tuple=allow_tuple, allow_list=allow_list)
    if has_label:
        label = _convert_list_tuple(combine(label_list),
                                    allow_tuple=allow_tuple, allow_list=allow_list)
    else:
        label = None

    return data, label


# todo: this might be very slow
def to_sample(data):
    from bigdl.util.common import Sample
    data = check_type_and_convert(data, allow_list=True, allow_tuple=False)
    features = data["x"]
    length = features[0].shape[0]
    if "y" in data:
        labels = data["y"]
    else:
        labels = np.array([[-1] * length])

    for i in range(length):
        fs = [feat[i] for feat in features]
        ls = [l[i] for l in labels]
        if len(fs) == 1:
            fs = fs[0]
        if len(ls) == 1:
            ls = ls[0]
        yield Sample.from_ndarray(np.array(fs), np.array(ls))


def read_pd_hdfs_file_list(iterator, file_type, **kwargs):
    import pyarrow as pa
    fs = pa.hdfs.connect()
    dfs = []
    for x in iterator:
        with fs.open(x, 'rb') as f:
            df = read_pd_file(f, file_type, **kwargs)
            dfs.append(df)
    import pandas as pd
    return [pd.concat(dfs)]


def read_pd_s3_file_list(iterator, file_type, **kwargs):
    access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    dfs = []
    for x in iterator:
        path_parts = x.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = read_pd_file(obj['Body'], file_type, **kwargs)
        dfs.append(df)
    import pandas as pd
    return [pd.concat(dfs)]


def read_pd_file(path, file_type, **kwargs):
    import pandas as pd
    if file_type == "csv":
        df = pd.read_csv(path, **kwargs)
    elif file_type == "json":
        df = pd.read_json(path, **kwargs)
    else:
        raise Exception("Unsupported file type: %s. Only csv and json files are "
                        "supported for now" % file_type)
    return df


def get_class_name(obj):
    if obj.__class__.__module__ != 'builtins':
        return '.'.join([obj.__class__.__module__, obj.__class__.__name__])
    return obj.__class__.__name__


def _convert_list_tuple(data, allow_tuple, allow_list):
    if isinstance(data, list):
        if not allow_list and allow_tuple:
            return tuple(data)
    else:
        if not allow_tuple and allow_list:
            return list(data)
    return data


def process_spark_xshards(spark_xshards, num_workers):
    from zoo.orca.data.shard import RayXShards
    data = spark_xshards
    if data.num_partitions() != num_workers:
        data = data.repartition(num_workers)
    ray_xshards = RayXShards.from_spark_xshards(data)
    return ray_xshards


def index_data(x, i):
    if isinstance(x, np.ndarray):
        return x[i]
    elif isinstance(x, dict):
        res = {}
        for k, v in x.items():
            res[k] = v[i]
        return res
    elif isinstance(x, tuple):
        return tuple(item[i] for item in x)
    elif isinstance(x, list):
        return [item[i] for item in x]
    else:
        raise ValueError(
            "data should be an ndarray, a dict of ndarrays, a tuple of ndarrays"
            " or a list of ndarrays, please check your input")


def get_size(x):
    if isinstance(x, np.ndarray):
        return len(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            return len(v)
    elif isinstance(x, tuple) or isinstance(x, list):
        return len(x[0])
    else:
        raise ValueError(
            "data should be an ndarray, a dict of ndarrays, a tuple of ndarrays"
            " or a list of ndarrays, please check your input")

class ArrowBatchBuilder:

    def __init__(self, schema):
        self.schema = schema
        self.data = [[] for i in range(len(self.schema))]

    def append_record(self, row):
        for idx, field in enumerate(self.schema):
            value = row[field.name]
            self.data[idx].append(value)

    def to_batch(self):
        batch = pa.record_batch(self.data, schema=self.schema)
        return batch

    def reset(self):
        for l in self.data:
            l.clear()


def deserialize_using_pa_stream(pybytes):
    # deserialize a list of rows from bytes
    buff = pa.py_buffer(pybytes)
    reader = pa.ipc.open_stream(buff)
    batches = [batch for batch in reader]
    return batches


def serialize_using_pa_stream(rows, schema):
    # serialize a list of rows to bytes
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, schema)
    arrow_batch_builder = ArrowBatchBuilder(schema)
    for row in rows:
        arrow_batch_builder.append_record(row)
    writer.write_batch(arrow_batch_builder.to_batch())
    writer.close()
    buf = sink.getvalue()
    return buf.to_pybytes()


def write_to_ray_python_client(idx, partition, redis_address, ip2port, schema):
    """
    used in mapwithPartitionIndex in pyspark process.
    create a client in pyspark process for each partition
    :param idx: partition index
    :param partition: partition content
    :param redis_address: ray redis address
    :param ip2port: {ip: port for each ray actor}
    :param schema: df schema
    """
    from multiprocessing.connection import Client
    from pyspark.sql.types import to_arrow_schema

    # find ip that ray used by connecting to redis.
    redis_host, redis_port = redis_address.split(":")
    ip = get_node_ip(redis_host)
    port = ip2port[ip]
    address = ("localhost", port)
    pa_schema = to_arrow_schema(schema)

    import time
    counter = 0
    serialize_time = 0.0
    sending_time = 0.0
    total_bytes = 0
    with Client(address) as conn:
        conn.send(idx)
        batch = []
        for item in partition:
            batch.append(item)
            # serialize for every 500k records
            if len(batch) == 500000:
                start = time.time()
                pybytes = serialize_using_pa_stream(batch, pa_schema)
                end1 = time.time()
                total_bytes += len(pybytes)
                serialize_time += end1 - start
                conn.send(pybytes)
                end2 = time.time()
                sending_time += end2 - end1
                counter += 500000
                batch.clear()
        # serialize for the last batch with less than 500k records
        if len(batch) > 0:
            start = time.time()
            pybytes = serialize_using_pa_stream(batch, pa_schema)
            total_bytes += len(pybytes)
            end1 = time.time()
            conn.send(pybytes)
            end2 = time.time()
            serialize_time += end1 - start
            sending_time += end2 - end1
            counter += len(batch)
        # use None as end indicator
        conn.send(None)
        print(
            f"total serilaize time {serialize_time}, total send time {sending_time}, total records "
            f"{counter}, total bytes {total_bytes}")

    return [counter]