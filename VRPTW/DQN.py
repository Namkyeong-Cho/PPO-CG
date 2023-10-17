import glob
import json
import os
import dill
import numpy as np
from copy import deepcopy
import random
from env import VRP
from net import BipartiteGNN
from agents import Agent
import tensorflow as tf


class DQNAgent(Agent):
    '''
    '''
    def __init__(self, env, **parameters):
        print("args DQN Agent :", parameters['parameters'])
        print("DQN Agent called")
        print("*" * 100)
        
        embedding_size = parameters['parameters']['embedding_size']
        cons_num_features = parameters['parameters']['cons_num_features']
        vars_num_features = parameters['parameters']['vars_num_features']
        capacity = parameters['parameters']['capacity']
        learning_rate = parameters['parameters']['lr']
        batch_size = parameters['parameters']['batch_size']
        epochs = parameters['parameters']['epochs']
                    #    hidden_dim,
                    #    batch_size,
                    #    epochs,
                    #    embedding_size,
                    #    cons_num_features,
                    #    vars_num_features,
                    #    learning_rate,
                    #    seed_

        super(DQNAgent, self).__init__(env, capacity, 
                                       parameters=parameters['parameters'])
        self.embedding_size = embedding_size 
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.lr = learning_rate

        self.behavior_Q = BipartiteGNN(embedding_size = self.embedding_size, cons_num_features = self.cons_num_features, 
        vars_num_features = self.vars_num_features, learning_rate = self.lr, parameters=parameters['parameters'])

        self.target_Q = BipartiteGNN(embedding_size = self.embedding_size, cons_num_features = self.cons_num_features, 
        vars_num_features = self.vars_num_features, learning_rate = self.lr, parameters=parameters['parameters'])
        self._update_target_Q()
        
        self.batch_size = batch_size 
        self.epochs = epochs



        
    def _update_target_Q(self):
       
        '''
        '''
        self.target_Q.set_weights(deepcopy(self.behavior_Q.variables))
        # self.target_Q = self.behavior_Q.clone()
        
    
    ## s is the super s0, A is the list containing all actions
    def policy(self, action_info, s, epsilon = None):

        total_added, Actions = action_info
        Q_s = self.behavior_Q(s)
        Q_s_for_action = Q_s[-total_added::]

        rand_value = np.random.random()
        if epsilon is not None and rand_value < epsilon:
            return random.choice(list(Actions))
        else:
            idx = int(np.argmax(Q_s_for_action))
            return Actions[idx]


    ## s is the super s0, A is the list containing all actions
    ### need action info 0 and total 1 (total_1 to get max Q_1, action_info_0 to get update index)
    def get_max(self,  total_1, s):
        Q_s = self.target_Q.call(s)
        Q_s_for_action = Q_s[-total_1::]
        return np.max(Q_s_for_action)

    ## this method is used to get target in _learn_from_memory function
    ## select the max number from last few items from Q_matrix for each row, based on last_index_list
    
    def _learn_from_memory(self, gamma, learning_rate):

        ## trans_pieces is a list of transitions
        trans_pieces = self.sample(self.batch_size)  # Get transition data
        states_0 = np.vstack([x.s0 for x in trans_pieces]) # as s0 is a list, so vstack
        actions_0 = np.array([x.a0 for x in trans_pieces]) 
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_dones = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])
        action_info = np.vstack([x.action_info_0 for x in trans_pieces])
        totals_0 = np.vstack([x.total_0 for x in trans_pieces])
        totals_1 = np.vstack([x.total_1 for x in trans_pieces])

        y_batch = []
        for i in range(len(states_0)):
          
            ### get the index of action that is taken at s0
            acts_0 = action_info[i][1]
            act_0 = list(actions_0[i])

            idx = 0
            for act in acts_0:
                if act==act_0:
                    break
                idx+=1

            y = self.target_Q.call(states_0[i]).numpy()
            #### set the non action terms to be 0 
            y[0:-totals_0[i][0]] = 0

            if is_dones[i]:
                Q_target = reward_1[i]
            else:
                ### the number of actions for state 1 is used to get Q_target
                Q_max = self.get_max(totals_1[i][0], states_1[i])
                Q_target = reward_1[i] + gamma * Q_max

            y[-totals_0[i][0]+idx] = Q_target
            
            y_batch.append(np.asarray(y))

        y_batch= np.asarray(y_batch)
        X_batch = states_0
        
        loss = self.behavior_Q.train_or_test(X_batch, y_batch, totals_0, actions_0, action_info, True)
        # print("The loss is,", loss)
        self._update_target_Q()
      
        return loss

                         
    def learning(self, name_file, check_point_path=None, check_point=None):
      epsilon = self.epsilon
      decaying_epsilon = self.decay_epsilon
      gamma = self.gamma
      learning_rate = self.lr
      self.display = True
      display = self.display
      min_epsilon = self.min_epsilon
      min_epsilon_ratio = self.min_epsilon_ratio
      model_index = self.model_index
      total_time,  episode_reward, num_episode = 0,0,0
      total_times, episode_rewards, num_episodes,history = [], [], [], []

      schedule = np.load(name_file)
      max_episode_num = self.parameters['parameters']['max_episode_num']
      os.makedirs('./outputs/train_results/DQN/model_check_points/', exist_ok=True)
      os.makedirs('./outputs/train_results/DQN/log_data/', exist_ok=True)
      # check training process
      target_model_path = './outputs/train_results/DQN/target_DQN_Model_final_model.pth'
      behavior_model_path = './outputs/train_results/DQN/behavior_DQN_Model_final_model.pth'
      training_done = os.path.isfile(target_model_path) and os.path.isfile(behavior_model_path)
      if training_done:
        print("Training is completed for DQN")
        assert False
      max_iter = 0
      # if model training is not complete get max iter num
      if glob.glob('./outputs/train_results/DQN/model_check_points/*') != []:
        for x in glob.glob('./outputs/train_results/DQN/model_check_points/*'):
            iter = int(x.split('_')[-1].split('.')[0])
            max_iter = max(max_iter, iter)
            # load model
        with open(F"./outputs/train_results/DQN/log_data/{max_iter}_agent_experience.dill", "rb") as dill_file:
            # Write the dictionary to the file in JSON format
            self.experience = dill.load(dill_file)
        self.behavior_Q.restore_state(f'./outputs/train_results/DQN/model_check_points/behavior_DQN_Model_{max_iter}.pth') 
        self.target_Q.restore_state(f'./outputs/train_results/DQN/model_check_points/target_DQN_Model_{max_iter}.pth') 
            
      for i in range(max_episode_num):
          if i+1 <= max_iter:
              continue
          print(f'{i+1}th episode traing begins...')
          print('-'*100)
          if epsilon is None:
              epsilon = 1e-10
          elif decaying_epsilon:
              #epsilon = 1.0 / (1 + num_episode)
              epsilon = self._decayed_epsilon(cur_episode = num_episode+1,
                                              min_epsilon = min_epsilon,
                                              max_epsilon = epsilon,
                                              target_episode = int(max_episode_num * min_epsilon_ratio))
          

         
          #### read_file    

          # schedule = np.load("/content/drive/MyDrive/ICML_VRP/schedule.npy")
          n,problem_name = schedule[i]

          try:
              VRP_instance = VRP(problem_name,int(n))
          except:
              print(problem_name +" is not found, so skip")
              continue
        #   VRP_instance = VRP(problem_name,int(n))

          too_long = VRP_instance.initialize()[2]

          if too_long:
              print("Skip instance {} in episode {}".format(problem_name,i))
              continue
                    
          training_time, number_of_step, average_loss, episode_reward = self.learning_method(VRP_instance, \
                gamma = gamma, learning_rate = learning_rate, epsilon = epsilon, display = display)
          num_episode += 1


          if num_episode %1==0:
                target_model_save_name = 'target_DQN_Model_'+str(i+1)+'.pth'
                path_model_check = F'./outputs/train_results/DQN/model_check_points/{target_model_save_name}'
                self.target_Q.save_state(path_model_check)

                behavior_model_save_name = 'behavior_DQN_Model_'+str(i+1)+'.pth'
                path_model_check = F'./outputs/train_results/DQN/model_check_points/{behavior_model_save_name}'
                self.behavior_Q.save_state(path_model_check)

          if i+1 == max_episode_num:                
                target_model_save_name = 'target_DQN_Model_final_model.pth'
                path_model_check = F'./outputs/train_results/DQN/model_check_points/{target_model_save_name}'
                self.target_Q.save_state(path_model_check)

                behavior_model_save_name = 'behavior_DQN_Model_final_model.pth'
                path_model_check = F'./outputs/train_results/DQN/model_check_points/{behavior_model_save_name}'
                self.behavior_Q.save_state(path_model_check)
            
          each_episolde_data ={
                'epsidoe_num': num_episode,
                'instance_num':i+1,
                'training_time':training_time,
                'episode_reward':episode_reward,
                'episode_loss' : average_loss,
                'num_of_step' : number_of_step,
                'instance_name': problem_name,
                'num_of_n' : VRP_instance.nf
            }
            
          with open(F"./outputs/train_results/DQN/log_data/{i+1}_agent_experience.dill", "wb") as dill_file:
                # Write the dictionary to the file in JSON format
                dill.dump(self.experience, dill_file)

          with open(F"./outputs/train_results/DQN/log_data/{i+1}_episode_data.dill", "wb") as dill_file:
                # Write the dictionary to the file in JSON format
                dill.dump(each_episolde_data, dill_file)
          if tf.is_tensor(each_episolde_data['episode_loss']):
            each_episolde_data['episode_loss']=each_episolde_data['episode_loss'].numpy()
          for key, val in each_episolde_data.items():
                each_episolde_data[key]=str(val)
          with open(F"./outputs/train_results/DQN/log_data/{i+1}_episode_data.json", "w") as json_file:
                # Write the dictionary to the file in JSON format
                json.dump(each_episolde_data, json_file, indent=4)
    
          # total_time += time_in_episode
    #       total_times.append(time_in_episode)
    #       episode_rewards.append(episode_reward)
    #       num_episodes.append(num_episode)
    #       history.append(VRP_instance.objVal_history[-1])
    #       print("Episode: " + str(i) + " takes " + str(time_in_episode) +" steps with obj "+str(VRP_instance.objVal_history[-1]))
    #   model_save_name = 'Model.pt'
    #   path_model = 'Saved_results/{model_save_name}'
    #   self.target_Q.save_state(path_model)

    
 
      return  total_times, episode_rewards, num_episodes



    #### the learning code
    def learning_method(self, instance, gamma, learning_rate, epsilon, 
                        display):
        import time
        starting_time = time.time()
        epochs = self.epochs
        ###########
        self.env = instance
        self.S = self.get_aug_state()
        
        time_in_episode, total_reward = 0, 0
        is_done = False
        average_loss = 0
        number_of_step  = 0
        while not is_done:
            number_of_step += 1
            s0_aug = self.S[0]
            action_info = self.S[1]
            ### a0 is selected based on behavior_Q
            # print(action_info[1])
            if len(action_info[1]) == 0:
                ## if no available actions,end this episode
                break;
            
            a0 = self.policy(action_info, s0_aug, epsilon)
            print("Added route is {} out of {}".format(a0,len(action_info[1])))

            s1_augmented, r, is_done, total_reward = self.act(a0)

            if self.total_trans > self.batch_size:
                for e in range(epochs):
                    average_loss += self._learn_from_memory(gamma, learning_rate)
                # loss/=epochs
            # s0 = s1
            time_in_episode += 1

        average_loss /= (time_in_episode*epochs)
        print("The loss is,",average_loss)
        if display:
            print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon, average_loss,self.experience.last_episode))
        training_time = time.time() - starting_time
        return training_time, number_of_step, average_loss, total_reward 
