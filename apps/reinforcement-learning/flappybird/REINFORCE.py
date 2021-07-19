
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import sys
import os


work_space = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#work_space = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_space+"/game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import json
import os

from bigdl.optim.optimizer import *
from bigdl.nn.criterion import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Dense, Dropout, Activation,Flatten,Convolution2D
import tensorflow as tf
sc = get_spark_context()


def to_RDD(X, y):
    return sc.parallelize(X).zip(sc.parallelize(y)).map(
            lambda x: Sample.from_ndarray(x[0],x[1]))

def normalize(advantages, smallEps=1e-8):
    return (advantages - advantages.mean())/(advantages.std() + smallEps)

ACTION_PER_FRAME = 1
IMAGE_ROWS, IMAGE_COLS = 80, 80
IMAGE_CHANNELS = 4
ACTION_SIZE = 2
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 3000000
BATCH_SIZE = 1 # every how many eposides do a parameter update
GAMMA = 0.99
r_reward_moving_average = 0
average_step_record = list()

# the abstraction of birdagent
class BirdAgent:
    def __init__(self, state_size, action_size, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.model = self._build_mode()

    def _build_mode(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                                input_shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)))  # 80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
        
        # get the 1 * 2 output represent each action's probability
        model.add(Activation('softmax'))
        return model

    # sample the action from the predict of the model
    def action_sampler(self, out):
        # return 1 if result > np.random.random() else 0
        return np.random.choice([0, 1], p=out)

    def act(self, state, epsilon = 0.01):
        explore = False
        random_num = 1
        # random_num = random.random()
        action = np.zeros(2)
        if random_num <= epsilon:
            explore = True
            # ramdomly select an action
            action_index = random.randrange(ACTION_SIZE)
            action[action_index] = 1
            print("*********** Random Action *********** : ", action_index)
            return action,action,explore

        else:
            if isinstance(state, np.ndarray):
                features = to_sample_rdd(state, np.zeros([state.shape[0]]))
            out = self.model.predict(features)
            #print("out type",out)
            out = out.collect()
            print("out result ",out)
            
            if self.action_sampler(np.squeeze(out)) == 0:
                action[0] = 1
            else:
                action[1] = 1
            return np.squeeze(out),action,explore


def to_sample_rdd(x, y, numSlices=None):
    """
    Conver x and y into RDD[Sample]
    :param x: ndarray and the first dimension should be batch
    :param y: ndarray and the first dimension should be batch
    :param numSlices:
    :return:
    """
    from bigdl.util.common import Sample
    x_rdd = sc.parallelize(x, numSlices)
    y_rdd = sc.parallelize(y, numSlices)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


# the abstraction of rollouts
class RollOuts():
    def __init__(self, next_states, actions, rewards, logprobs, gradients, total_steps):
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.steps = total_steps
        self.logprobs = logprobs
        self.gradients = gradients
        self.total_rewards = np.sum(rewards)


    def get_summary(self):
        return {" total_reward ": self.total_rewards,
                " total_steps ": self.steps}


    def prepare_target(self):
        result = []
        for action, adv in list(zip(self.actions, self.advs)):
            adv = adv * action
            print('action and advantages : ',action, adv)
            result.append(adv)
        return np.array(result)

# the abstraction of experience memory
class ExperienceStroe:
    def __init__(self):
        self.rollouts = []

    def add_rollout(self, next_states, actions, rewards,logprobs, gradients, total_steps):
        self.rollouts.append(RollOuts(next_states, actions, rewards, logprobs, gradients,total_steps))

    def num_experiences(self):
        return len(self.rollouts)

    def get_range(self, start, end):
        return self.rollouts[start:end]

    def reset(self):
        self.rollouts = []


def stats_summary(rollouts, records, verbose=True):
    rollout_rewards = np.array([rollout.total_rewards for rollout in rollouts])
    print("reward mean %s" % (rollout_rewards.mean()))
    print("reward std %s" % (rollout_rewards.std()))
    print("reward max %s" % (rollout_rewards.max()))
    records.append([rollout_rewards.mean(), rollout_rewards.std()])



# calculate running reward
def cal_reward(rollouts, gamma, discounted=True):
    for rollout in rollouts:
        rewards = rollout.rewards
        r_reward = []
        running_reward = 0
        for reward in rewards[::-1]:
            # discounted reward
            # reward = (-1) * reward
            if discounted == True:
                running_reward = gamma * running_reward + reward
            else:
                running_reward = running_reward + reward
            r_reward.append(running_reward)
        rollout.r_rewards = np.squeeze(np.array(np.vstack(r_reward[::-1])))


def cal_advantage_moving_average(rollouts):
    global r_reward_moving_average
    for rollout in rollouts:
        advs = np.zeros([rollout.rewards.shape[0]])
        i = 0
        for r_reward in rollout.r_rewards:
            print("running_rewards and r_reward_moving_average",r_reward ,r_reward_moving_average)
            # calculate moving average
            r_reward_moving_average = 0.9 * r_reward_moving_average + (1-0.9) * r_reward
            # correct the bias
            advs[i] = r_reward - r_reward_moving_average
            i += 1
            # rollout.advs = normalize(advs)   
            rollout.advs=advs
        print("advantage ",advs)


def copy_r_reward_to_advs(rollouts):
    for rollout in rollouts:
        rollout.advs = rollout.r_rewards 
        print("running reward ", rollout.r_rewards)


# calculate the advantage = reward - expected reward at this time step
def cal_advantage(rollouts):
    max_steps = max(rollout.rewards.shape[0] for rollout in rollouts)
    for rollout in rollouts:
        rollout.r_rewards = np.pad(rollout.r_rewards, (0, max_steps - rollout.r_rewards.shape[0]),'constant')
    baselines = np.mean(np.vstack([rollout.r_rewards for rollout in rollouts]), axis=0)
    for rollout in rollouts:
        rollout.advs = rollout.r_rewards - baselines
        rollout.advs = rollout.advs[:len(rollout.r_rewards)]
        rollout.r_rewards = rollout.r_rewards[:len(rollout.rewards)]


def play_game(agent, render=False):
    # 1. initialize
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # used to save the actions, should be a 1*2 nparray and sum(action) should be 1
    actions = np.zeros([1,2])
    # used to save the rewards
    rewards = np.array([])
    # get the first state by doing nothing and preprocess the image to 80x80x4
  
    gradients = np.zeros([1, 2])
    # should be (1*2) nparray and each col represent the probability to chose that action
    logprobs = np.array([1, 2])

    do_nothing = np.zeros(ACTION_SIZE)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # preprocess the first frame
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    # rescale to 0-1
    x_t = x_t / 255.0

    # 80 * 80 * 4
    state = np.stack((x_t, x_t, x_t, x_t), axis=2)
    print (state.shape)
    # In Keras, need to reshape 1 * 80 * 80 * 4
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])  # 1*80*80*4
    # the observations will be batch_size * 80 * 80 * 4
    observations = np.zeros([1, state.shape[1], state.shape[2], state.shape[3]])
    for step in range(1, 500):
        # next state's shape, should be (1*80*80*4)
        
        # get the state list
        observations = np.vstack((observations, state))
        # predict the action

        logprob,action,explore = agent.act(state)
        
        
        actions = np.vstack((actions, action))
        logprobs = np.vstack((logprobs ,logprob))
        gradients = np.vstack((gradients , action.astype('float32') - logprob))
        # use the predicted action to determine the next state
        x_t1_colored, reward, terminal = game_state.frame_step(action)
        # print ('reward ',reward)
        # rgb to gray and rescale
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        # rescale to 0-1
        x_t1 = x_t1 / 255
        # update state
        x_t1 = x_t1.reshape(1,x_t1.shape[0],x_t1.shape[1],1)
        state = np.append(x_t1,state[:,:,:,:3],axis=3)
        rewards = np.append(rewards, reward)

        print('living steps : ',observations.shape[0])
        # print('state size ',state.shape)
        # print('action size : ',action.shape)
        # print('action : ',action)
        # print('reward : ',reward)
        # print('terminal : ',terminal)
        if terminal or step == 498:
            break
    return observations[1:], actions[1:], rewards , logprobs[1:], gradients[1:], step


def play_n_games(agent, history, n=4, verbose=True):
    start_eps = history.num_experiences()
    total_step = 0
    for i in range(n):
        observations, actions, rewards, logprobs, gradients, step = play_game(agent=agent)
        history.add_rollout(observations, actions, rewards, logprobs, gradients, step)
        total_step += step
    end_eps = history.num_experiences()
    return start_eps, end_eps, total_step


def learn(agent, rollouts):
    cal_reward(rollouts, agent.gamma)
    cal_advantage_moving_average(rollouts)
    
    #copy_r_reward_to_advs(rollouts)
    # cal_advantage(rollouts)
    X_batch = np.zeros([1, 80, 80, 4])
    Y_batch = np.array([0,0])
    for rollout in rollouts:
        X_batch = np.vstack((X_batch, rollout.next_states))
        # Y_batch: the label of the training should be [[action,adv]]
        Y_batch = np.vstack((Y_batch, rollout.prepare_target()))
    X_batch = X_batch[1:]
    Y_batch = Y_batch[1:]
    # Y_batch[:,1] = normalize(Y_batch[:,1])

    # Y_batch[:, 1] = normalize(y_batch[:, 1])
    # prepare to train the model
    print('X_batch size : ',X_batch.shape)
    print('Y_batch size : ',Y_batch.shape)
    rdd_sample = to_RDD(X_batch, Y_batch)
    batch_size = X_batch.shape[0] - X_batch.shape[0]%8
    print ("using batch_size = ",batch_size)
    
    
    rmsprop = RMSprop(learningrate=1e-6, decayrate=0.9)
    agent.model.compile(rmsprop,PGCriterion())
    agent.model.fit(rdd_sample,batch_size=8, nb_epoch=1)
    return agent.model
    
    
#     optimizer = Optimizer(model=agent.model,
#                                   training_rdd=rdd_sample,
#                                   criterion=PGCriterion(),
#                                   optim_method= RMSprop(learningrate=0.005),
#                                   end_trigger=MaxIteration(1),
#                                   batch_size=batch_size)
#     #else:
#         #optimizer.set_traindata(training_rdd=rdd_sample, batch_size=batch_size)
#         #optimizer.set_criterion(RFPGCriterion())
#     agent.model = optimizer.optimize()

import timeit
state_size = (1, 80, 80, 4)
action_size = ACTION_SIZE
agent = BirdAgent(state_size, action_size)
history = ExperienceStroe()
record = []
exe_times = []
history.reset()
train_start = 0
played_steps = 0
start_of_play = timeit.default_timer()
t=0
while True:
	t+=1
    # history is a list of rollouts
    start_eps, end_eps, steps = play_n_games(agent, history, n=BATCH_SIZE)
    played_steps += steps
    train_end = end_eps
    end_of_play = timeit.default_timer()
    print
    "*************************"
    rollouts = history.get_range(start_eps, end_eps)
    stats_summary(rollouts, record)
    if (played_steps > BATCH_SIZE * 34):
        print("used in training: ", start_eps, "-", end_eps)
        print("num of total steps in training ", played_steps)
        start_of_train = timeit.default_timer()
        rollouts = history.get_range(train_start, train_end)
        print("training begin")
        model = learn(agent, rollouts)
        if (t%100 == 0):
            model.saveModel("/model/model"+str(t)+".bigdl","/model/mode"+str(t)+"l.bin",True)

        end_of_train = timeit.default_timer()
        exe_time_game_play = end_of_play - start_of_play
        train_time_game_paly = end_of_train - start_of_train
        exe_times.append([exe_time_game_play, train_time_game_paly])
        train_start = end_eps
        train_end = end_eps
        played_steps = 0
        start_of_play = timeit.default_timer()

