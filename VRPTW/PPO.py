from __future__ import print_function
import copy
import glob

import time
import numpy as np
import torch
import dill
import json
from env import VRP

from net import BipartiteGNN
from utility import debug
from agents import Agent
from copy import deepcopy
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config
# Enable numpy-related methods
np_config.enable_numpy_behavior()
################################## set device ##################################
def choose_index_with_probability(probabilities):
    chosen_index = tf.random.categorical(tf.math.log([probabilities]), num_samples=1)
    return chosen_index[0, 0]
################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_with_action =[]
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_with_action[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Actor_Critic(BipartiteGNN):
    '''
    Initialization of the different modules and attributes
    Attributes :
    - embedding_size : Embedding size for the intermediate layers of the neural networks
    - cons_num_features : Number of constraint features, the constraints data matrix expected has the shape (None,cons_num_features)
    - vars_num_features : Number of variable features, the variables data matrix expected has the shape (None,vars_num_features)
    - learning_rate : Optimizer learning rate
    - activation : Activation function used in the neurons
    - initializer : Weights initializer
    '''

    def __init__(self, embedding_size=32, cons_num_features=2,
                 vars_num_features=9, learning_rate=1e-3,
                 activation=K.activations.relu, initializer=K.initializers.Orthogonal,
                 **parameters):
        
        vars_num_features =  parameters['parameters']['vars_num_features']
        cons_num_features =  parameters['parameters']['cons_num_features']
        learning_rate =  parameters['parameters']['actor_lr']
        
        super(Actor_Critic, self).__init__(
            embedding_size=32, cons_num_features=cons_num_features,
            vars_num_features=vars_num_features, learning_rate=learning_rate,
            activation=K.activations.relu, initializer=K.initializers.Orthogonal,
            **parameters
        )

        self.actor_module = K.Sequential([
            K.layers.Dense(units=embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.BatchNormalization(),
            K.layers.Dense(units=embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.BatchNormalization(),
            K.layers.Dense(units=1, activation=self.activation, kernel_initializer=self.initializer)
        ])
        self.critic_module = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.BatchNormalization(),
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.BatchNormalization(),
            K.layers.Dense(units=1, activation=self.activation, kernel_initializer=self.initializer)
        ])
        self.actor_lr = parameters['parameters']['actor_lr']
        self.critic_lr = parameters['parameters']['critic_lr']
        self.optimizer = tf.optimizers.Adam(learning_rate=self.actor_lr)
        self.build_for_PPO()
        self.action_dim = parameters['parameters']['action_dim']
    '''
    Build function, sets the input shapes. Called during initialization
    '''
    # Actor And Critic
    def build_for_PPO(self):
        print("PPO build")
        self.cons_embedding.build([None, self.cons_num_features])
        self.var_embedding.build([None, self.vars_num_features])
        self.join_features_NN.build([None, self.embedding_size * 2])
        self.cons_representation_NN.build([None, self.embedding_size * 2])
        self.vars_representation_NN.build([None, self.embedding_size * 2])
        self.critic_module.build([None, self.embedding_size])
        self.actor_module.build([None, self.embedding_size])
        self.built = True
        del self.output_module
        self.variables_topological_order = [v.name for v in self.trainable_variables]
        self.compile(optimizer=self.optimizer)
    '''
    Main function taking as an input a tuple containing the three matrices :
    - cons_features : Matrix of constraints features, shape : (None, cons_num_features)
    - edge_indices : Edge indices linking constraints<->variables, shape : (2, None)
    - vars_features : Matrix of variables features, shape : (None, vars_num_features)

    Output : logit vector for the variables nodes, shape (None,1)
    '''

    def get_feature_from_state(self, state):
        cons_features, edge_indices, vars_features = state
        cons_features = self.cons_embedding(cons_features)
        vars_features = self.var_embedding(vars_features)
        # ==== First Pass : Variables -> Constraints ====
        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        cons_features,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                    ### change this number to edge weights (patterns)
                ],1)
        )

        # Aggregation step
        output_cons = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[0], axis=1),
            shape=[cons_features.shape[0], self.embedding_size]
        )
        # Constraints representations update
        output_cons = self.cons_representation_NN(tf.concat([output_cons,cons_features],1))



        # ==== Second Pass : Constraints -> Variables ====
        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        output_cons,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                ],1)
        )
        # print("joint_features :" , joint_features)
        # Aggregation step
        output_vars = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[1], axis=1),
            shape=[vars_features.shape[0], self.embedding_size]
        )
        # print("output_vars :", output_vars)
        # Variables representations update
        output_vars = self.vars_representation_NN(tf.concat([output_vars,vars_features],1))
        # output = self.output_module(output_vars)
        return output_vars
    def select_action(self,S_without_action, S_with_action):
        feat1 = self.get_feature_from_state(S_with_action[0])
        total_added = S_with_action[1][0]
        output = self.actor_module(feat1[-total_added:])
        # print("output of actor : ", output)
        output = output / tf.reduce_sum(output)
        # idx = choose_index_with_probability(output.reshape(-1))
        idx = int(np.argmax(output))
        action = S_with_action[1][1][idx]
        # print("action : ", action, idx)
        feat2 = self.get_feature_from_state(S_without_action[0])
        state_val = self.critic_module(feat2)
        state_val = tf.reduce_mean(state_val).reshape(1)
        # print("state_val: ", state_val)
        action_logprob = tf.math.log(output[idx]) 

        return action, action_logprob, state_val

    def evaluate(self, S_with_action, S_without_action):
        feat1 = self.get_feature_from_state(S_with_action[0])
        total_added = S_with_action[1][0]
        output = self.actor_module(feat1[-total_added:])

        output = output / tf.reduce_sum(output)
        idx = choose_index_with_probability(output.reshape(-1))
        # print("idx: " , idx)
        action_logprob = tf.math.log(output[idx])
        # print("action_logprob : ", action_logprob)
        # assert False
        # dist = tfp.distributions.Categorical(probs=action_logprob)
        # Compute the entropy
        # entropy = dist.entropy()

        feat2 = self.get_feature_from_state(S_without_action[0])
        state_val = self.critic_module(feat2)
        state_val = tf.reduce_mean(state_val).reshape(1)
        return action_logprob, state_val
        #, entropy


    '''
    Training/Test function
    Input: 
    - data : a batch of data, type : tf.data.Dataset
    - train: boolean, True if function called for training (i.e., compute gradients and update weights),
                False if called for test
    Output:
    tuple(Loss, Accuracy, Recall, TNR) : Metrics
    '''

    def train_or_test(self, data, labels, totals_0, actions_0, action_info, train=False):
        mean_loss = 0

        batches_counter = 0

        ###########################################################
        ### how does this data(a batch) relates to transition data?
        ###########################################################
        for batch in data:
            cons_features, edge_indices, vars_features = batch
            input_tuple = (cons_features, edge_indices, vars_features)

            total_0 = totals_0[batches_counter]

            label = labels[batches_counter]

            # action = actions_0[batches_counter].tolist()

            # all_actions = action_info[batches_counter][1].tolist()

            # print(all_actions)
            # act_index = all_actions.index(action) ## used for getting the correct loss -> only conunts the loss of the selected actions

            # When called train=True, compute gradient and update weights
            if train:
                with tf.GradientTape() as tape:
                    # Get logits from the bipartite GNN model

                    ########
                    ######### may need to change to self.call?
                    logits = self.call(input_tuple)
                    # print(total_0)
                    label[0:-total_0[0]] = logits[0:-total_0[0]]  ## do not count the loss from the nodes already in the basis
                    # print(abel[act_index])
                    # print()
                    # print(logits[-total_0[0]:-1])
                    # print()
                    loss = tf.keras.metrics.mean_squared_error(label, logits)  ## should not be mean_squared_error as it's then scaled down by number of nodes
                    loss = (loss * label.shape[0]) / total_0[0]  ## this is a quick fix, as there are far less action nodes compared to
                    ## again, do we calculate the loss using the nodes we are not selecting?

                    # print(loss)
                # Compute gradient and update weights
                grads = tape.gradient(target=loss, sources=self.trainable_variablea)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variablea))
            # If no optimizer instance set, no training is performed, give outputs and metrics only
            else:
                logits = self.call(input_tuple)
                loss = tf.keras.metrics.mean_squared_error(label, logits)

            ## these are for classification
            # prediction = tf.round(tf.nn.sigmoid(logits))
            # correct_pred = tf.equal(prediction, label)
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            loss = tf.reduce_mean(loss)

            # Batch loss, accuracy, confusion matrix
            mean_loss += loss
            batches_counter += 1
            # confusion_mat += confusion_matrix(labels, prediction)

        # Batch average loss
        mean_loss /= batches_counter
        return mean_loss


class PPOAgent(Agent):
    '''
    '''

    def __init__(self, env, **parameters):
        print("args PPO Agent :", parameters['parameters'])
        print("PPO Agent called")
        print("*" * 100)

        embedding_size = parameters['parameters']['embedding_size']
        cons_num_features = parameters['parameters']['cons_num_features']
        vars_num_features = parameters['parameters']['vars_num_features']
        capacity = parameters['parameters']['capacity']
        critic_lr = parameters['parameters']['critic_lr']
        actor_lr = parameters['parameters']['actor_lr']
        batch_size = parameters['parameters']['batch_size']
        epochs = parameters['parameters']['epochs']
        eps_clip = parameters['parameters']['eps_clip']

        super(PPOAgent, self).__init__(env, capacity, parameters=parameters['parameters'])
        
        self.embedding_size = embedding_size
        # self.hidden_dim = hidden_dim
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.buffer = RolloutBuffer()
        # self._update_target_Q()
        self.policy = Actor_Critic(parameters=parameters['parameters'])
        self.old_policy = Actor_Critic(parameters=parameters['parameters'])
        self.old_policy.set_weights(deepcopy(self.policy.variables))
        self.batch_size = batch_size
        self.epochs = epochs
        self.eps_clip = eps_clip

    def update(self):
        # Monte Carlo estimate of returns
        # Monte Carlo estimate of returns
        # print("In the update!")
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + ( discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards


        ## trans_pieces is a list of transitions

        old_states = np.vstack(self.buffer.states)  # as s0 is a list, so vstackv
        old_states_with_action = np.vstack(self.buffer.states_with_action)  # as s0 is a list, so vstackv
        old_actions = np.array(self.buffer.actions)
        old_logprobs = np.array(self.buffer.logprobs)
        old_state_values = np.array(self.buffer.state_values)
        y_batch = []
        rewards = tf.constant(rewards, dtype=tf.float32).reshape(len(old_states),-1)
        old_state_values = tf.constant(old_state_values, dtype=tf.float32).reshape(len(old_states),-1)
        rewards = (rewards - rewards.mean()) / (tf.math.reduce_std(rewards)  + 1e-7)
        advantages = rewards - old_state_values
        # self.epochs = 1
        with tf.GradientTape() as tape:
            logprobs, state_values, dists = [], [], []
            for i in range(len(old_states)):
                logprob, state_value = self.policy.evaluate( old_states_with_action[i],old_states[i])
                ### get the index of action that is taken at s0
                logprobs.append(logprob)
                state_values.append(state_value)
                # dists.append(dist)

            state_values =tf.concat(state_values, axis=0).reshape(len(old_states),-1 )
            old_logprobs = tf.concat(old_logprobs, axis=0).reshape(len(old_states),-1 )
            logprobs = tf.concat(logprobs, axis=0).reshape(len(old_states),-1 )

            ratios = tf.math.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = tf.reduce_mean(-tf.math.minimum(surr1, surr2) + 0.5 * tf.keras.metrics.mean_squared_error(state_values, rewards))
            # loss = -tf.math.minimum(surr1, surr2) + 0.5 * tf.keras.metrics.mean_squared_error(state_values, rewards)
            # loss = tf.reduce_mean(loss)
        grads = tape.gradient(target=loss, sources=self.policy.trainable_variables)
        # print("grads : ", grads,loss)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        # print("self.policy:", self.policy.trainable_variables)
        # self.policy.save_state('polciy.pth')
        # with open('grads.dill', 'wb') as f:
        #     dill.dump(grads,f)
        # print("grads : ", grads)

        self.old_policy.set_weights(deepcopy(self.policy.variables))
        self.buffer.clear()
        return loss
    #
    def learning(self, name_file):
        print("PPO LEARNING!!!")
        epsilon = self.epsilon
        decaying_epsilon = self.decay_epsilon
        gamma = self.gamma
        # learning_rate = self.lr

        max_episode_num = self.max_episode_num
        self.display = True
        display = self.display
        min_epsilon = self.min_epsilon
        min_epsilon_ratio = self.min_epsilon_ratio
        model_index = self.model_index
        total_time, episode_reward, num_episode = 0, 0, 0
        total_times, episode_rewards, num_episodes = [], [], []

        # max_episode_num now set to be 480 as there are 480 instances uploaded
        max_episode_num = self.parameters['parameters']['max_episode_num']
        schedule = np.load(name_file)
        os.makedirs('./outputs/train_results/PPO/model_check_points/', exist_ok=True)
        os.makedirs('./outputs/train_results/PPO/log_data/', exist_ok=True)
        
        # check training process
        old_polict_model_path = './outputs/train_results/PPO/old_policy_PPO_Model_final_model.pth'
        policy_model_path = './outputs/train_results/PPO/policy_PPO_Model_final_model.pth'

        training_done = os.path.isfile(old_polict_model_path) and os.path.isfile(policy_model_path)
        if training_done:
            print("Training is completed for PPO")
            assert False
        max_iter = 0
        # if model training is not complete get max iter num
        if glob.glob('./outputs/train_results/PPO/model_check_points/*') != []:
            for x in glob.glob('./outputs/train_results/PPO/model_check_points/*'):
                iter = int(x.split('_')[-1].split('.')[0])
                max_iter = max(max_iter, iter)
            # load model
            self.policy.restore_state(f'./outputs/train_results/PPO/model_check_points/policy_PPO_Model_{max_iter}.pth') 
            self.old_policy.restore_state(f'./outputs/train_results/PPO/model_check_points/old_policy_PPO_Model_{max_iter}.pth') 
        for i in range(max_episode_num):
            if i+1 <= max_iter:
                continue   
            print(f'{i+1}th episode traing begins...')
            print('-'*100)
            start_time = time.time()
            number_of_step = 0
            #### read_file
            n,problem_name = schedule[i]
            ## TEMP CODE ##
            json_file_path = f'./outputs/train_results/PPO_original_method/log_data/{i+1}_episode_data.json'
            if os.path.exists(json_file_path) is False:
                continue
            try:
              VRP_instance = VRP(problem_name,int(n))
            except:
              print(problem_name +" is not found, so skip")
              continue
            too_long = VRP_instance.initialize(test_or_not=False)[2]

            if too_long:
                print("Skip instance {} in episode {}".format(problem_name,i))
                continue
            if VRP_instance == "not found":
                print("########### NOT FOUND ###############")
                continue
            total_loss = 0
            for epoch in range(self.epochs):
                
                if epsilon is None:
                    epsilon = 1e-10
                elif decaying_epsilon:
                    # epsilon = 1.0 / (1 + num_episode)
                    epsilon = self._decayed_epsilon(cur_episode=num_episode + 1,
                                                    min_epsilon=min_epsilon,
                                                    max_epsilon=0.05,
                                                    target_episode=int(max_episode_num * min_epsilon_ratio))

                
                VRP_instance.initialize(test_or_not=True)
                self.env = VRP_instance
                self.S = self.get_aug_state(include_available_action=False)

                num_of_steps, total_reward = 0, 0
                is_done = False

                if epoch == 0:  # first time iteration
                    while not is_done:
                        S_without_action = self.get_aug_state(include_available_action=False)
                        S_with_action = self.get_aug_state(include_available_action=True)
                        if len(S_with_action[1][1]) == 0:
                            break
                        action, action_logprob, state_val = self.old_policy.select_action(
                                                                S_without_action=S_without_action,
                                                                S_with_action=S_with_action)
                        print("Added route is {} out of {}".format(action,len(S_with_action[1][1])))
                        self.buffer.states.append(S_without_action)
                        self.buffer.states_with_action.append(S_with_action)
                        self.buffer.actions.append(action)
                        self.buffer.logprobs.append(action_logprob)
                        self.buffer.state_values.append(state_val)

                        reward, is_done = self.env.step(action, test_or_not=True)
                        self.buffer.rewards.append(reward)
                        self.buffer.is_terminals.append(is_done)
                        total_reward += reward
                        number_of_step += 1
                    self.temp_buffer = copy.deepcopy(self.buffer)
                else:
                    self.buffer = copy.deepcopy(self.temp_buffer)
                # (TO Do) check point, weight save
                loss = self.update()
                total_loss += loss
                episode_reward += total_reward

            num_episode += 1
            training_time = time.time() - start_time
            episode_reward /= self.epochs
            # number_of_step /= self.epochs
            total_loss /= self.epochs
            print("Episode: " + str(num_episode) + " takes " + str(training_time) +" seconds")
            if num_episode%1==0:
                old_policiy_model_save_name = 'old_policy_PPO_Model_'+str(i+1)+'.pth'
                path_model_check = F'./outputs/train_results/PPO/model_check_points/{old_policiy_model_save_name}'
                self.old_policy.save_state(path_model_check)

                policy_model_save_name = 'policy_PPO_Model_'+str(i+1)+'.pth'
                path_model_check = F'./outputs/train_results/PPO/model_check_points/{policy_model_save_name}'
                self.policy.save_state(path_model_check)

            if i+1 == max_episode_num:                
                old_policiy_model_save_name = 'old_policy_PPO_Model_final_model.pth'
                path_model_check = F'./outputs/train_results/PPO/model_check_points/{old_policiy_model_save_name}'
                self.old_policy.save_state(path_model_check)

                policy_model_save_name = 'policy_PPO_Model_final_model.pth'
                path_model_check = F'./outputs/train_results/PPO/model_check_points/{policy_model_save_name}'
                self.policy.save_state(path_model_check)

            each_episolde_data ={
                'epsidoe_num': num_episode,
                'instance_num':i+1,
                'training_time':training_time,
                'episode_reward':episode_reward,
                'num_of_step' : number_of_step,
                'instance_name': problem_name,
                'num_of_n' : VRP_instance.n,
                'total_loss' : total_loss,
            }
            with open(F"./outputs/train_results/PPO/log_data/{i+1}_episode_data.dill", "wb") as dill_file:
                # Write the dictionary to the file in JSON format
                dill.dump(each_episolde_data, dill_file)

            for key, val in each_episolde_data.items():
                each_episolde_data[key]=str(val)
            with open(F"./outputs/train_results/PPO/log_data/{i+1}_episode_data.json", "w") as json_file:
                # Write the dictionary to the file in JSON format
                json.dump(each_episolde_data, json_file, indent=4)
        return total_times, episode_rewards, num_episodes