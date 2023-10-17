#### The cutting stock object is the environment
# %pip install -i https://pypi.gurobi.com gurobipy
import argparse
import json
from gurobipy import GRB
import numpy as np
import random
from copy import deepcopy
import gurobipy as gp
import math
from collections import namedtuple
from typing import List
from tqdm import tqdm
from DQN import DQNAgent
import dill
import os
import time
from PPO import PPOAgent
from test import general_compare
from parameters import PARAMETERS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Get main arguments')
parser.add_argument('--RL_algorithm', type=str, choices=['DQN', 'PPO', 'no_RL'],
                    help='what rl method do you use?')
parser.add_argument('--model_name', type = str, default='result',
                    help='Name of trained model')
parser.add_argument('--test', action='store_true', help='Are we testing or traing?')
parser.add_argument('--train', action='store_true', help='Are we testing or traing?')
args = parser.parse_args()
if args.RL_algorithm != 'no_RL':
    with open('./inputs/parameters.json') as f:
        # Load the JSON data from the file
        parameters = json.load(f)[args.RL_algorithm]

# Parameters = PARAMETERS()
# random.seed(Parameters.seed)
# np.random.seed(Parameters.seed)
# epsilon_ = Parameters.epsilon
# decaying_epsilon_ = Parameters.decaying_epsilon
# gamma_ = Parameters.gamma
# alpha_ = Parameters.alpha_obj_weight
# max_episode_num_ = Parameters.max_episode_num
# min_epsilon_ = Parameters.min_epsilon
# min_epsilon_ratio_ = Parameters.min_epsilon_ratio
# capacity_ =  Parameters.capacity
# hidden_dim_ = Parameters.hidden_dim
# batch_size_ = Parameters.batch_size
# epochs_ = Parameters.epochs
# embedding_size_ = Parameters.embedding_size
# cons_num_features_ = Parameters.cons_num_features
# vars_num_features_ = Parameters.vars_num_features
# learning_rate_ = Parameters.lr
# model_index_ = Parameters.model_index
# seed_fix = Parameters.seed
# display_ = False

if args.train:
    start_time = time.time()
    print("Training in process")
    if args.RL_algorithm == 'DQN':
        os.makedirs('./outputs/train_results/DQN/', exist_ok=True)
        with open('./outputs/train_results/DQN/args.dill', 'wb') as wb:
            dill.dump(args, wb)
        agent = DQNAgent(env = None, parameters=parameters)
        schedule_train_name = "train_test/schedule.npy"
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        total_times, episode_rewards, num_episodes = agent.learning(schedule_train_name)

    elif args.RL_algorithm == 'PPO':
        start_time = time.time()
        agent = PPOAgent(env = None, parameters=parameters)
        schedule_train_name = "train_test/schedule.npy"
        os.makedirs('./outputs/train_results/PPO/', exist_ok=True)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        with open('./outputs/train_results/PPO/args.dill', 'wb') as wb:
            dill.dump(args, wb)
        total_times, episode_rewards, num_episodes = agent.learning(schedule_train_name)

    print(F"Training Takes {time.time()-start_time} seconds in total")


if args.test:
    print("Testing in process")
    if args.RL_algorithm == 'DQN':
        agent = DQNAgent(env=None, parameters=parameters)
        DATA = general_compare(agent, 'DQN')
    elif args.RL_algorithm == 'PPO':
        agent = PPOAgent(env=None, parameters=parameters)
        DATA = general_compare(agent,'PPO')
    else:
        DATA = general_compare(None, 'no_RL')
    # print("DATA  :  ", DATA)
    print("*" * 100)
# ### training and saving the data for plotting and model weights (weights and data are saved inside .learning)


# DQN = DQNAgent(env = None, capacity = capacity_, 
# hidden_dim = hidden_dim_, batch_size = batch_size_,
#  epochs = epochs_, embedding_size = embedding_size_, 
# 			   cons_num_features = cons_num_features_,
#  vars_num_features = vars_num_features_, 
# learning_rate = learning_rate_, seed_ = seed_fix)



# time_start = time.time()
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)     
# total_times, episode_rewards, num_episodes =  DQN.learning(epsilon = epsilon_, decaying_epsilon = decaying_epsilon_, gamma = gamma_, 
#                 learning_rate = learning_rate_, max_episode_num = max_episode_num_, display = display_, min_epsilon = min_epsilon_, min_epsilon_ratio = min_epsilon_ratio_, model_index = model_index_)    

# time_ends = time.time()