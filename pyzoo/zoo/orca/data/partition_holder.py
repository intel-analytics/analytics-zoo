import uuid

import ray
from zoo.ray import RayContext
from zoo.orca.data.utils import deserialize_using_pa_stream, write_to_ray_python_client
from zoo.orca.learn.pytorch.utils import find_free_port


class PartitionHolder:
    # for ray actor
    def __init__(self):
        import queue
        from concurrent.futures import ThreadPoolExecutor
        self.data_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=44)
        self.data_dict = {}

    def start_data_receiver(self):
        ip = ray.services.get_node_ip_address()
        port = self._start_server()
        self._start_deserialize()
        return ip, port

    def _start_server(self):
        import threading
        from multiprocessing.connection import Listener

        def handle_conn(conn, queue):
            idx = conn.recv()
            item = conn.recv()
            while item is not None:
                queue.put((idx, item))
                item = conn.recv()
            conn.close()

        def run_server(listener, queue, executor):
            while True:
                # return until client connected. (call Client(addr))
                conn = listener.accept()
                # submit is async.
                executor.submit(handle_conn, conn, queue)

        # todo: handle port being used after find_free_port before listen
        port = find_free_port()
        address = ("localhost", port)
        listener = Listener(address)
        server_thread = threading.Thread(target=run_server,
                                         args=(listener, self.data_queue, self.executor))
        server_thread.start()
        return port

    def _start_deserialize(self):
        import threading

        def deserialize(queue, data_dict):
            import pyarrow as pa
            import time
            start = time.time()
            counter = 0
            while True:
                data = queue.get()
                if data is None:
                    end = time.time()
                    for idx, data in data_dict.items():
                        table = pa.Table.from_batches(batches=data)
                        data_dict[idx] = table
                    print(f"done deserializing, using {end - start}s")
                    break
                else:
                    idx, bs = data
                    if idx in data_dict:
                        data_dict[idx].extend(deserialize_using_pa_stream(bs))
                    else:
                        data_dict[idx] = deserialize_using_pa_stream(bs)
                    counter += 1
                    print(f"done with {counter} batches, queue size {queue.qsize()}")

        self.deser_thread = threading.Thread(target=deserialize,
                                             args=(self.data_queue, self.data_dict))
        self.deser_thread.start()

    def done_with_sending(self):
        self.executor.shutdown()
        self.data_queue.put(None)
        self.deser_thread.join()
        total = 0
        for idx, data in self.data_dict.items():
            total += len(data)

        return total

    def to_torch_train_loader(self, feature_cols, label_cols, batch_size):
        # TODO: convert self.data_dict to torch loader
        pass


def spark_rdd_python_server(df):
    def get_ray_node_ip():
        driver_ip = ray.services.get_node_ip_address()
        ray_nodes = []
        for key, value in resources.items():
            if key.startswith("node:"):
                # if running in cluster, filter out driver ip
                if not (not ray_ctx.is_local and key == f"node:{driver_ip}"):
                    ray_nodes.append(key)
        return ray_nodes

    ray_ctx = RayContext.get()
    address = ray_ctx.redis_address
    password = ray_ctx.redis_password

    num_cores = ray_ctx.ray_node_cpu_cores

    # universal unique id, for rdd
    uuid_str = str(uuid.uuid4())
    resources = ray.cluster_resources()
    nodes = get_ray_node_ip()

    partition_holders = [ray.remote(num_cpus=0, resources={node: 1e-4})(PartitionHolder).remote()
                         for node in nodes]
    ray.get([holder.start_data_receiver.remote() for holder in partition_holders])
    ip_ports = ray.get([holder.get_ip_port.remote() for holder in partition_holders])

    schema = df.schema

    id_ip_counts = df.rdd.mapPartitionsWithIndex(lambda idx, part: write_to_ray_python_client(
        idx, part, address, dict(ip_ports), schema)).collect()

    id_ips = [(i, ip) for i, ip, _ in id_ip_counts]

    print(f"writing records {sum([count for _, _, count in id_ip_counts])}")
    result = ray.get(
        [holder.done_with_sending.remote() for holder in partition_holders])
    print(f"getting records {sum(result)}")
    return uuid_str, dict(id_ips), partition_holders


if __name__ == "__main__":
    pass
    # from zoo.orca import init_orca_context
    # from pyspark.sql import SparkSession
    # import numpy as np
    #
    # sc = init_orca_context(cores=8)
    # spark = SparkSession(sc)
    # rdd = sc.parallelize([tuple(i + j for i in range(3)) for j in range(80)])
    # df = rdd.toDF(["feature", "label", "c"])
    # spark_rdd_python_server(df=df)
