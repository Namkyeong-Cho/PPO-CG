
import random
import numpy as np

from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import VRP
from read_data import addRoutesToMaster


from utility import Experience, Transition





class Agent(object):
  '''Base Class of Agent
  '''
  def __init__(self, initial_env=None, capacity = 10000, **parameters):
      self.env = initial_env # the evironment would be one cutting stock object
      ## add the env and available action will be added in the learning_method 
      # self.A = self.env.available_action
      self.parameters=parameters
      self.epsilon = parameters['parameters']['epsilon']
      self.gamma = parameters['parameters']['gamma']
      self.max_episode_num = parameters['parameters']['max_episode_num']
      self.min_epsilon = parameters['parameters']['min_epsilon']
      self.min_epsilon_ratio = parameters['parameters']['min_epsilon_ratio']
      self.model_index = parameters['parameters']['model_index']
      self.decay_epsilon = parameters['parameters']['decaying_epsilon']
      print("Agent intiated")
      self.A = []
      self.experience = Experience(capacity = capacity)
      # S record the current super state for the agent
      # self.S = self.get_aug_state()   

      self.S = []


  ## get augmented state from the current environment
  def get_aug_state(self, include_available_action=True):
      
      # actions,reduced_costs = deepcopy(self.env.available_action)
      if include_available_action:
        actions,reduced_costs = deepcopy(self.env.available_action)
      else:
          actions, reduced_costs = (), ()
    #   reduced_costs = action_info[0]
      
      total_added = len(actions)

      patterns = self.env.routes[:]

      is_action = np.asarray([0]*len(patterns))

      patterns.extend(actions)



      col_num = len(patterns)
      cons_num = self.env.n
      column_features = []
      cons_features = []
      edge_indices = [[],[]]




      # RC = self.env.RC[:]
      # RC = np.append(RC,reduced_costs)


      
      ###################################################
      MatA = deepcopy(self.env.A)
      cost_c = deepcopy(self.env.c)

      newMat = np.zeros((self.env.n, len(actions)))
      newCosts = np.zeros(len(actions))
      addRoutesToMaster(actions, newMat, newCosts, self.env.d) 

      # routes += newRoutes
      MatA = np.c_[MatA, newMat]

      In_Cons_Num = np.count_nonzero(MatA, axis=0)
      ###################################################


      ColumnSol_Val = self.env.ColumnSol_Val[:]
      ColumnSol_Val = np.append(ColumnSol_Val, np.zeros(total_added))
      cost_c = np.append(cost_c, newCosts)

      stay_in = self.env.stay_in[:]
      stay_in = np.append(stay_in,np.zeros(total_added))
      stay_out = self.env.stay_out[:]
      stay_out = np.append(stay_out,np.zeros(total_added))
      just_left = self.env.just_left[:]
      just_left = np.append(just_left,np.zeros(total_added))
      just_enter = self.env.just_enter[:]
      just_enter = np.append(just_enter,np.zeros(total_added))

      is_action = np.append(is_action,np.ones(total_added))
      


      #### constraint features, also augment all actions
      Shadow_Price = self.env.Shadow_Price[:]
      In_Cols_Num = np.count_nonzero(MatA, axis=1)

      # In_Cols_Num = self.env.In_Cols_Num[:]
      # for action in actions:
      #     non_zero = np.nonzero(action)
      #     for idx in non_zero:
      #         In_Cols_Num[idx]+=1
    
    

      Shadow_Price = np.asarray(Shadow_Price).reshape(-1, 1)
      In_Cons_Num = np.asarray(In_Cons_Num).reshape(-1, 1)
      In_Cols_Num = np.asarray(In_Cols_Num).reshape(-1, 1)
      ColumnSol_Val = np.asarray(ColumnSol_Val).reshape(-1, 1)
      cost_c = np.asarray(cost_c).reshape(-1, 1)
      stay_in = np.asarray(stay_in).reshape(-1, 1)
      stay_out = np.asarray(stay_out).reshape(-1, 1)


      from sklearn.preprocessing import MinMaxScaler
      # from sklearn.preprocessing import StandardScalar


      Scaler_SP = MinMaxScaler()
      Scaler_SP.fit(Shadow_Price)
      Shadow_Price = Scaler_SP.transform(Shadow_Price)
      Scaler_IConsN = MinMaxScaler()
      Scaler_IConsN.fit(In_Cons_Num)
      In_Cons_Num = Scaler_IConsN.transform(In_Cons_Num)
      Scaler_IColsN = MinMaxScaler()
      Scaler_IColsN.fit(In_Cols_Num)
      In_Cols_Num = Scaler_IColsN.transform(In_Cols_Num)
      Scaler_CSV = MinMaxScaler()
      Scaler_CSV.fit(ColumnSol_Val)
      ColumnSol_Val = Scaler_CSV.transform(ColumnSol_Val)
      Scaler_W = MinMaxScaler()
      Scaler_W.fit(cost_c)
      cost_c = Scaler_W.transform(cost_c)

      Scaler_si = MinMaxScaler()
      Scaler_si.fit(stay_in)
      stay_in = Scaler_si.transform(stay_in)

      Scaler_out = MinMaxScaler()
      Scaler_out.fit(stay_out)
      stay_out = Scaler_out.transform(stay_out)


      Shadow_Price = list(Shadow_Price.T[0])
      In_Cons_Num = list(In_Cons_Num.T[0])
      In_Cols_Num = list(In_Cols_Num.T[0])
      ColumnSol_Val = list(ColumnSol_Val.T[0])
      cost_c = list(cost_c.T[0])
      stay_in = list(stay_in.T[0])
      stay_out = list(stay_out.T[0])


      ### constraint nodes
      for j in range(cons_num):
          con_feat = []
          con_feat.append(Shadow_Price[j])
          con_feat.append(In_Cols_Num[j])
          cons_features.append(con_feat)

      
      ### normalize here for each information
      for i in range(col_num):
          col_feat = []
          col_feat.append(In_Cons_Num[i])
          col_feat.append(ColumnSol_Val[i])
          col_feat.append(cost_c[i])
          col_feat.append(stay_in[i])
          col_feat.append(stay_out[i])
          col_feat.append(just_left[i])
          col_feat.append(just_enter[i])
          col_feat.append(is_action[i])

          column_features.append(col_feat)
      
      ## get edges going
      for m in range(len(MatA[0])):
          for n in range(len(MatA)):
              if MatA[n][m]!=0:
                  # then mth column is connected to nth cons
                  edge_indices[0].append(m)
                  edge_indices[1].append(n)

      edge_indices = np.asarray(edge_indices)
      edge_indices[[0, 1]] = edge_indices[[1, 0]]


      cons_features=np.asarray(cons_features)
      column_features=np.asarray(column_features)

      ## need this total_added for reading the Q values, need actions to select onne pattern after read Q values
      aug_state, action_info = ((cons_features, edge_indices, column_features),(total_added,actions))
      return aug_state, action_info

  def policy(self):

      return random.choice(self.A)
      # return random.sample(self.A, k=1)[0]
  
  def perform_policy(self, s, Q = None, epsilon = 0.05):
      action = self.policy()
      return action


  def act(self, a0):
      ## get the current super state 
      s0_augmented, action_info_0 = self.S
      total_0 = deepcopy(action_info_0[0])
      # print(s0_augmented)

      ## step change the environnment, update all the information used for agent to construct state
      r, is_done = self.env.step(a0)

      s1_augmented, action_info_1 = self.get_aug_state()
      total_1 = action_info_1[0]
      trans = Transition(s0_augmented, a0, r, is_done, s1_augmented, action_info_0, total_0, total_1)
      total_reward = self.experience.push(trans)
      self.S = s1_augmented, action_info_1

      return s1_augmented, r, is_done, total_reward

  def learning_method(self,VRP_instance, gamma = 0.9, alpha = 1e-3, epsilon = 0.05):
      #self.state = self.env.reset()
      ## initialize an environment


      self.env = VRP_instance


      self.S = self.get_aug_state()

      ### initialize before calling
      # self.env.initialize()

      self.A = self.env.available_action[0]

      s0 = self.S
      a0 = self.perform_policy(s0, epsilon)
      time_in_episode, total_reward = 0, 0
      is_done = False
      while not is_done:
          # act also update self.S
          s1, r1, is_done, total_reward = self.act(a0)
          self.A = self.env.available_action[0]
          # if self.A == []:
          #     break;
          a1 = self.perform_policy(s1, epsilon)
          s0, a0 = s1, a1
          time_in_episode += 1

          # actions,reduced_costs = deepcopy(self.env.available_action)

      return time_in_episode, total_reward  
    

  def _decayed_epsilon(self,cur_episode: int, 
                            min_epsilon: float, 
                            max_epsilon: float, 
                            target_episode: int) -> float: 
      print("*"*100)
      print(min_epsilon)
      print(max_epsilon)
      print("*"*100)
      slope = (min_epsilon - max_epsilon) / (target_episode)
      intercept = max_epsilon
      return max(min_epsilon, slope * cur_episode + intercept)        
      
  def learning(self, name_file, check_point_path=None, check_point=None):
      pass

  def sample(self, batch_size = 32):
      return self.experience.sample(batch_size)

  @property
  def total_trans(self):
      return self.experience.total_trans
  
  def last_episode_detail(self):
      self.experience.last_episode.print_detail()

