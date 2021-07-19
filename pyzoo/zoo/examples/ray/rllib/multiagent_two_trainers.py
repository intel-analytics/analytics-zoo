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
from __future__ import division
from __future__ import print_function
"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two trainers (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse
import gym
import os

import ray
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray.util.raycontext import RayContext
os.environ["LANG"] = "C.UTF-8"

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                    " configuration folder. Otherwise, turn on local mode.")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.hadoop_conf:
        slave_num = 2
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name="ray36",
            num_executor=slave_num,
            executor_cores=28,
            executor_memory="10g",
            driver_memory="2g",
            driver_cores=4,
            extra_executor_memory_for_ray="30g")
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory="25g")
    else:
        sc = init_spark_on_local(cores=4)
        ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    # Simple environment with 4 independent cartpole entities
    register_env("multi_cartpole", lambda _: MultiCartpole(4))
    single_env = gym.make("CartPole-v0")
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {
        "ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
        "dqn_policy": (DQNTFPolicy, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"

    ppo_trainer = PPOTrainer(
        env="multi_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "NoFilter",
        })

    dqn_trainer = DQNTrainer(
        env="multi_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "gamma": 0.95,
            "n_step": 3,
        })

    # disable DQN exploration when used by the PPO trainer
    ppo_trainer.workers.foreach_worker(
        lambda ev: ev.for_policy(
            lambda pi: pi.set_epsilon(0.0), policy_id="dqn_policy"))

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))
    ray_ctx.stop()
