import dill
import matplotlib
import json
matplotlib.use('Agg')
from test import general_compare
import random
import numpy as np
import argparse
import warnings
from DQN import DQNAgent
from PPO import PPOAgent
import os
import time
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

#
# if args.RL_algorithm == 'DQN':
#     agent = DQNAgent(env = None, parameters=parameters)
#     schedule_train_name = "inputs/Name_files/Scheduled_train.txt"
#     total_times, episode_rewards, num_episodes = agent.learning(schedule_train_name)
# elif args.RL_algorithm == 'PPO':
#     agent = PPOAgent(env=None, parameters=parameters)
# elif args.RL_algorithm == 'no-RL':
#     raise NotImplementedError
# else:
#     raise NotImplementedError

if args.train:
    start_time = time.time()
    print("Training in process")
    if args.RL_algorithm == 'DQN':
        os.makedirs('./outputs/train_results/DQN/', exist_ok=True)
        with open('./outputs/train_results/DQN/args.dill', 'wb') as wb:
            dill.dump(args, wb)
        agent = DQNAgent(env = None, parameters=parameters)
        schedule_train_name = "inputs/Name_files/Scheduled_train.txt"
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        total_times, episode_rewards, num_episodes = agent.learning(schedule_train_name)

    elif args.RL_algorithm == 'PPO':
        start_time = time.time()
        agent = PPOAgent(env = None, parameters=parameters)
        schedule_train_name = "inputs/Name_files/Scheduled_train.txt"
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
        DATA = general_compare(agent, 0, 200, 'DQN')
        DATA = general_compare(agent, 0, 750, 'DQN')
    elif args.RL_algorithm == 'PPO':
        agent = PPOAgent(env=None, parameters=parameters)
        DATA = general_compare(agent, 0, 200)
        DATA = general_compare(agent, 0, 750)
    else:
        DATA = general_compare(None, 0, 200, 'no_RL')
        DATA = general_compare(None, 0, 750, 'no_RL')
    # print("DATA  :  ", DATA)
    print("*" * 100)
    # except:
    #     print("Agent not defined")
