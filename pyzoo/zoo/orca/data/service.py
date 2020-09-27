import pyarrow as pa
import redis
import uuid
import time
import threading
import torch

class RedisQueue:
    def __init__(self, host=None, port=None, name="data_loader"):
        self.name = name
        if not host:
            host = "localhost"
        if not port:
            port = "6379"
        self.destroyed = False

        self.db = redis.StrictRedis(host=host,
                                    port=port, db=0, password="123456")

    def enqueue(self, data):
        """
        deprecated
        """
        if not self.destroyed:
            buf = pa.serialize(data).to_buffer()
            b = buf.to_pybytes()
            while not self.destroyed and self.db.llen(self.name) > 100:
                time.sleep(1)
            self.db.rpush(self.name, b)


    def dequeue(self):
        key, value = self.db.blpop(self.name)
        restored_data = pa.deserialize(value)
        return restored_data

    def destroy(self):
        self.destroyed = True
        # todo this may happen before the last enqueue
        self.db.delete(self.name)

class DataIter:

    """Iterator that counts upward forever."""

    def __init__(self, queue, length):
        self.queue = queue
        self.length = length
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.length:
            raise StopIteration()
        else:
            self.counter += 1
            result = self.queue.dequeue()
            # print(result)
            return result

class RedisQueueDataset(torch.utils.data.IterableDataset):

    def __init__(self, name, data_length):
        self.name = name
        self.queue = None
        self.data_length = data_length

    def __len__(self):
        return self.data_length

    def __iter__(self):
        if self.queue is None:
            self.queue = RedisQueue(name=self.name)

        return DataIter(self.queue, self.data_length)

def generate_data(data_loader, queue):

    while True:
        for data in data_loader:
            img, label = data
            if not queue.destroyed:
                queue.enqueue((img.numpy(), label.numpy()))
            else:
                return


class DataService:

    def __init__(self, data_loader, num_consumers=1):
        self.name = "data-queue-" + str(uuid.uuid4())
        self.data_loader = data_loader
        self.num_consumers = num_consumers
        self.queue = RedisQueue(name=self.name)
        self.thread = threading.Thread(target=generate_data, args=(self.data_loader, self.queue))
        self.thread.setDaemon(True)

    def make_distributed_data_loader(self, num_workers=0):
        data_loader = torch.utils.data.DataLoader(RedisQueueDataset(self.name,
                                                                    len(self.data_loader)//self.num_consumers),
                                                  batch_size=None, num_workers=num_workers)
        return data_loader

    def start(self):
        self.thread.start()

    def stop(self):
        self.queue.destroy()
        self.thread.join()