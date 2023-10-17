## where the functions used for testing are stored; 
## these functions are used for generating the data needed in  plotting_functions.py testing functions

import numpy as np
import matplotlib.pyplot as plt
from env import *
from DQN import *
from read_data import *
from gurobipy import GRB
import gurobipy as gp
import time



def follow_policy(model, action_info, s, model_type):
    '''DQN selects an action
    '''
    if model_type == 'DQN':
        total_added, Actions = action_info
        Q_s = model.target_Q(s)
        Q_s_for_action = Q_s[-total_added::]
        # rand_value = np.random.random()
        idx = int(np.argmax(Q_s_for_action))
        print('idx : ',idx)
        if len(Actions) == 0:
            return []
        return Actions[idx]
    elif model_type == 'PPO':
        S_without_action = model.get_aug_state(include_available_action=False)
        S_with_action = model.get_aug_state(include_available_action=True)
        if len(S_with_action[1][1]) == 0:
            action = []
        else:
            action, action_logprob, state_val = model.policy.select_action(S_without_action=S_without_action,
                                                                           S_with_action=S_with_action)
        return action


def general_compare(model, RL_algorithm='PPO'):
    import json
    with open('./inputs/parameters.json', 'r') as f:
        parameters = json.load(f)
    schedule_test = np.load('train_test/test.npy',allow_pickle=True)
     ################################## Greedy
        

    # print("names",names)
    Greedy = []
    Expert = []
    RL = []

    True_obj = []  ## do we put it here and have this for all instances ?

    print("#####################")
    print("Starts testing for model " + RL_algorithm)
    if RL_algorithm == 'PPO':
        old_policy_path_model = \
            './outputs/train_results/PPO/model_check_points/old_policy_PPO_Model_234.pth'
        policy_path_model = \
            './outputs/train_results/PPO/model_check_points/policy_PPO_Model_234.pth'
        model.policy.restore_state(policy_path_model)
        model.old_policy.restore_state(old_policy_path_model)
    elif RL_algorithm == 'DQN':
        behavior_DQN_model_path = './outputs/train_results/DQN/model_check_points/behavior_DQN_Model_final_model.pth'
        target_DQN_model_path = './outputs/train_results/DQN/model_check_points/target_DQN_Model_final_model.pth'
        model.target_Q.restore_state(target_DQN_model_path)
        model.behavior_Q.restore_state(behavior_DQN_model_path)

    for k in range(len(schedule_test)):
        save_file = f'./outputs/test_results/{RL_algorithm}/{RL_algorithm}_{k}_{len(schedule_test)}.dill'
        if os.path.exists(save_file):
            print("continue file :", save_file)
            continue
        # try:
        n,problem_name = schedule_test[k]
        if RL_algorithm =='no_RL':
        ## used for reading optimal values from excel files
        # name_of_instance = names[i]
        # opt_value_path = ''
        ################################## Greedy
            time1 = time.time()
            
            path_model = F'./outputs/test_results/{RL_algorithm}/'
            os.makedirs(path_model, exist_ok=True)
            ################################## Greedy
            try:
                VRP_instance = VRP(problem_name,int(n))
            except:
                print("Skip as not found")
                continue
            too_long = VRP_instance.initialize(test_or_not=True)[2]
            if too_long:
                print("skip instance "+problem_name, "too_long :", too_long)
            # print(" cut1 : ", cut1)
            # print("starts greedy")
            print(f"{k} th out of {len(schedule_test)} problem starts initializing  takes {time.time()-time1} seconds")
            is_done = False
            action_history =[]
            while True:
                if is_done:
                    break

                action = VRP_instance.available_action[0][0]
                reward, is_done = VRP_instance.step(action, test_or_not=True)

            history_opt_g = VRP_instance.objVal_history
            time2 = time.time()

            obj_greedy = history_opt_g[-1]
            steps_g = len(history_opt_g)
            Greedy.append((history_opt_g, len(history_opt_g), time2 - time1, obj_greedy))
            print("Greedy takes {} steps to reach obj {} with time {}".format(steps_g, obj_greedy, time2 - time1))

        else:
            time2 = time.time()
            print("starts RL")

            ###################################  RL
            print("problem_name: ", problem_name, int(n))
            try:
                VRP_instance = VRP(problem_name,int(n))
            except:
                print("Skip as not found")
                continue
            too_long = VRP_instance.initialize(test_or_not=True)[2]
            if too_long:
                print("skip instance "+ problem_name, "too_long :", too_long)
            print(f"{k} th out of {len(schedule_test)} problem starts initializing  takes {time.time()-time2} seconds")
            path_model = F'./outputs/test_results/{RL_algorithm}/'
            os.makedirs(path_model, exist_ok=True)
            # DQN_test.target_Q.restore_state(path_model)
            # DQN_test.behavior_Q.restore_state(path_model)
            model.env = VRP_instance
            before_get_aug_state = time.time()
            model.S = model.get_aug_state()
            after_get_aug_state = time.time()
            is_done = False
            action_history =[]
            while True:
                if is_done:
                    break

                action_info = model.S[1]
                s = model.S[0]
                action = follow_policy(model, action_info, s, RL_algorithm)
                print("Added route is {} out of {}".format(action,len(action_info[1])))
                if len(action) == 0:
                    break
                action_history.append(action)
                before_step = time.time()
                reward, is_done = VRP_instance.step(action, test_or_not=True)
                # print('each step takes :', time.time()-before_step, " seconds")
                model.S = model.get_aug_state()

            history_opt_rl = VRP_instance.objVal_history
            time3 = time.time()
            obj_RL = history_opt_rl[-1]
            steps_RL = len(history_opt_rl)
            # print("RL takes {} steps to reach obj {} with time {}".format(steps_RL, obj_RL, time3 - time2))
            # RL.append((history_opt_rl, len(history_opt_rl), time3 - time2, obj_RL))
            
        # ###### Compare with expert
     
        print()
        if RL_algorithm == 'no_RL':
            current_result = {
                'Greedy': {'history_opt_g': history_opt_g, 
                                            'history_length': steps_g, 
                                            'time': time2 - time1, 
                                            'objective': obj_greedy,
                                            # 'action_history':action_history
                            }
            }
            
        else:
            current_result = {
                          RL_algorithm: {'history_opt_rl': history_opt_rl, 
                                            'history_length': len(history_opt_rl), 
                                            'time': time3 - time2, 
                                            'objective': obj_RL,
                                            'action_history':action_history
                                            }
                          }

        with open(
                F'./outputs/test_results/{RL_algorithm}/{RL_algorithm}_{k}_{len(schedule_test)}.dill', 'wb') as wb:
            dill.dump(current_result, wb)
        print("{} steps out of {}".format(k, len(schedule_test)))
        print("#########")

    # complete_data = (Greedy, RL, Expert)
    complete_data = {
        "Greedy": Greedy,
        "RL": RL,
        # "Expert": Expert
    }

    # path = 'outputs/save_data/testing_data/'
    # np.save(path + 'testing_result_model_' + str(model_index) + "_size" + str(prob_size), complete_data)

    #### return three lists of tuple: (running time, steps)

    # return complete_data
