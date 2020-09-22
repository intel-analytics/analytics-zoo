import argparse
from os.path import join

import torch.nn as nn

from timm.models import create_model
from timm.optim import create_optimizer
from zoo.orca.learn.pytorch import Estimator

from zoo.orca import init_orca_context


def model_creator(config):
    model = create_model(
        "resnet50",
        pretrained=False,
        num_classes=1000,
        global_pool="avg")
    return model


def make_train_transform(args):
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def identity(x):
    return x


def worker_urls(urls):
    import webdataset as wds
    import sys
    result = wds.worker_urls(urls)
    print("worker_urls returning", len(result), "of", len(urls), "urls", file=sys.stderr)
    return result


def make_train_loader_wds(config):
    import webdataset as wds
    import torch
    print("=> using WebDataset loader")
    train_transform = make_train_transform(args)
    train_dataset = (
        wds.Dataset(config["train_url"], length=100, shard_selection=worker_urls)
            .shuffle(True)
            .decode("pil")
            .to_tuple("jpg;png;jpeg cls")
            .map_tuple(train_transform, identity)
            .batched(config["batch_size"])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, shuffle=False, num_workers=1,
    )
    return train_loader


def train_data_creator(config):
    # torch.manual_seed(args.seed + torch.distributed.get_rank())
    import os
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    train_dir = join(config["data_dir"], "train")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=1, pin_memory=False)

    return train_loader


def val_data_creator(config):
    # torch.manual_seed(args.seed + torch.distributed.get_rank())
    import os
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    val_dir = join(config["data_dir"], "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=1, pin_memory=False)

    return val_loader


def optimizer_creator(model, config):
    args = config["args"]
    return create_optimizer(args, model)


def loss_creator(config):
    # there should be more complicated logic here, but we don't support
    # separate train and eval losses yet
    return nn.CrossEntropyLoss()


def main(args):
    est = Estimator.from_torch(model=model_creator,
                               optimizer=optimizer_creator,
                               loss=loss_creator,
                               config={
                                   "batch_size": 32,
                                   "data_dir": "/home/yang/sources/datasets/imagenet-raw-image/imagenet-2012-small",
                                   "train_url": "pipe: python hdfs_client.py /yang/imagnet-webdataset/imagenet-train-000000.tar",
                                   "args": args,
                               },
                               use_tqdm=True, workers_per_node=4)
    est.fit(train_data_creator, epochs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to be used in the cluster. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--memory", type=str, default="10g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--workers_per_node", type=int, default=1,
                        help="The number of workers to run on each node")
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--sched", type=str, default="step")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmup-lr", type=float, default=0.0001)

    args = parser.parse_args()
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      num_nodes=args.num_nodes, memory=args.memory)

    main(args)
