# Model 3:  (300, 0.05, 0.9, 0.001)

class PARAMETERS(object):
    def __init__(self):
        self.seed = 5


        ## parameters about neural network
        self.lr = 1e-3  ##
        self.batch_size = 16
        self.hidden_dim = 32
        self.epochs = 5
        self.embedding_size = 32
        self.cons_num_features = 2
        self.vars_num_features = 8



        ## parameters of RL algorithm
        self.gamma = 0.99 ##
        self.epsilon = 0.2
        self.min_epsilon = 0.2
        self.min_epsilon_ratio = 0.99
        self.decaying_epsilon = False
        self.step_penalty = 1
        self.alpha_obj_weight = 5 ##
        self.action_pool_size = 10 
        self.max_episode_num = 447
        self.capacity = 20000 


        self.model_index = 3 #####

