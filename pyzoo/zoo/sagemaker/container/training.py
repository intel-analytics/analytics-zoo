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

from __future__ import absolute_import

import logging
import os
import shlex
import socket
import stat
import subprocess
import sys
import textwrap
import time

from retrying import retry
import sagemaker_containers.beta.framework as framework
from zoo.sagemaker.container.timeout import timeout

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SPARK_VERSION = "2.2.0"

MODEL_FILE_NAME = "model.npz"


def train(env, hyperparameters):
    """Runs Zoo training on a user supplied module in either a local or distributed
    SageMaker environment.

    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.
    """

    hosts = list(env.hosts)
    current_host = env.current_host

    # start nodes
    if len(hosts) > 1:
    # distributed env
        _start_ssh_daemon()

        if current_host == _get_master_host_name(hosts):
            _wait_for_worker_nodes_to_start_sshd(hosts)

            # start all nodes
            _start_all(current_host, hosts)

    # train
    if len(hosts) > 1:
        if current_host == _get_master_host_name(hosts):
            _run_training(env)
    else:
        _run_training(env)


def _run_training(env):
    logger.info('Invoking user training script.')

    checkpoint_dir = _get_checkpoint_dir(env)

    # first import user module
    user_module = framework.modules.import_module_from_s3(env.module_dir)
    import zoo.sagemaker.container.trainer
    trainer_class = zoo.sagemaker.container.trainer.Trainer
    trainer = trainer_class(customer_script=user_module,
                            input_channels=env.channel_dirs,
                            model_path=env.model_dir,
                            output_path=env.output_dir,
                            customer_params=env.hyperparameters,
                            checkpoint_path=checkpoint_dir
                            )

    trainer.train()


def _get_checkpoint_dir(env):
    if 'checkpoint_path' not in env.hyperparameters:
        return env.model_dir

    checkpoint_path = env.hyperparameters['checkpoint_path']

    job_name = env.job_name

    # If the checkpoint path already matches the format 'job_name/checkpoints', then we don't
    # need to worry about checkpoints from multiple training jobs being saved in the same location
    if job_name is None or checkpoint_path.endswith(os.path.join(job_name, 'checkpoints')):
        return checkpoint_path
    else:
        return os.path.join(checkpoint_path, job_name, 'checkpoints')


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _start_master():
    cmd = "/opt/work/spark-{}".format(_SPARK_VERSION) + "/sbin/start-master.sh"
    subprocess.check_call(cmd)


def _start_all(master_host, hosts):
    for host in hosts:
        if host != master_host:
            subprocess.check_call("echo %s >> /opt/work/spark-%s/conf/slave" % (host, _SPARK_VERSION))
    cmd = "/opt/work/spark-{}".format(_SPARK_VERSION) + "/sbin/start-all.sh"
    subprocess.check_call(cmd)


def _start_slave(master_host):
    cmd = "/opt/work/spark-{}".format(_SPARK_VERSION) + "/sbin/start-slaves.sh spark://{}:7077".format(master_host)
    subprocess.check_call(cmd)


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        logger.debug("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host %s", host)
        return True
    except socket.error:
        logger.debug("can't connect to host %s", host)
        return False


def main():
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)

    logger.setLevel(env.log_level)
    train(env, hyperparameters)


if __name__ == '__main__':
    main()
