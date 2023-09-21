import os
import argparse
import time
from datetime import datetime
import numpy as np
from itertools import count
from collections import namedtuple, deque
import pickle
import torch
import gym
import random
from Ippo_ik.ppo_tricks import PPO_tricks
from Ippo_ik.normalization import Normalization, RewardScaling
from Ippo_ik.replaybuffer import ReplayBuffer
from Ippo_ik.gym_env import ur5eGymEnv
import sys, getopt
import pybullet
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

os.environ['CUDA_VISIBLE_DEVICE']='1'
title = 'PyBullet UR5e robot'

class Ippo(object):
    def __init__(self, obj_pose, obs1_pos, obs2_pos, obs3_pos):
        super(Ippo, self).__init__()

        ################ PPO hyperparameters ################
        has_continuous_action_space = True  # continuous action space; else discrete
        self.max_ep_len = 100  # max timesteps in one episode
        max_episodes = int(1.5e4)
        max_training_timesteps = self.max_ep_len * max_episodes # break training loop if timeteps > max_training_timesteps
        action_std = 0.6
        max_action = 3.14
        hidden_width = 256
        lamda = 0.95
        batch_size = 4096
        max_train_steps = max_training_timesteps
        K_epochs = 20  # update policy for K epochs in one PPO update
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.0003  # learning rate for critic network
        random_seed = 1  # set random seed if required (0 = no random seed)
        #####################################################

        ################ Pybullet parameters ################
        render = False # render the environment
        repeat = 100 # repeat action 100
        task = 0 # task to learn: 0 move, 1 pick-up, 2 drop
        randObjPos = True # fixed object position to pick up
        simgrip = True # simulated gripper
        self.lp = 0.1 # Testing threshold

        #####################################################

        env_name = title
        self.env = ur5eGymEnv(renders=render, maxSteps=self.max_ep_len,
                actionRepeat=repeat, task=task, randObjPos=randObjPos,
                simulatedGripper=simgrip, learning_param=self.lp)

        self.env.seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        state_dim = 25

        if has_continuous_action_space:
            action_dim = 6
        else:
            action_dim = self.env.action_space.n

        #################### checkpointing ###################
        run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

        directory = "Ippo_ik/PPO_Tricks_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        actor_checkpoint_path = directory + "PPO_Tricks_actor_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        critic_checkpoint_path = directory + "PPO_Tricks_critic_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        ######################################################

        ################# PPO_tricks Setting ################
        use_reward_norm = False
        use_reward_scaling = False
        use_state_norm = False
        policy_dist = "Gaussian" # Gaussian or Beta
        replay_buffer = ReplayBuffer(batch_size, state_dim, action_dim)

        # initialize a PPO agent
        self.ppo_agent = PPO_tricks(batch_size, policy_dist, state_dim, action_dim, max_action, hidden_width, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, max_train_steps, action_std)
        state_norm = Normalization(state_dim)  # Trick 2:state normalization

        if use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=gamma)

        self.ppo_agent.actor_load(actor_checkpoint_path)
        self.ppo_agent.critic_load(critic_checkpoint_path)

    def ik(self, obj_pose, obs1_pos, obs2_pos, obs3_pos):
        state = self.env.reset(obj_pose, obs1_pos, obs2_pos, obs3_pos)
        for t in range(1, self.max_ep_len + 1):
            action, a_logprob = self.ppo_agent.choose_action(state, 0, 1)
            state_, reward, done, _ = self.env.step(action, obj_pose, obs1_pos, obs2_pos, obs3_pos, self.lp)
            state = state_
            if done:
                return action













