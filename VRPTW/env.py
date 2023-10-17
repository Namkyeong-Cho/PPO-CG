from time import process_time
import os
import time
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from heuristic_algorithm.compute_route import initializePathsWithImpact
from heuristic_algorithm.mp_sub_problem import reduceTimeWindows
from read_data import addRoutesToMaster, createDistanceMatrix, readData


class VRP(object):
    def __init__(self, INSTANCE_NAME, n): ## input the name of that instance

        # each instance corresponds to self.state = self.env.reset() in learning_method in agent.py

        # state: curent graph (connection, current column nodes, current constraint node and their features)
        #### static info: problem defination, same info used for initialization this instance

        Kdim, Q, x, y, q, a, b = readData(INSTANCE_NAME, n)
        
        #### fixed info
        self.n = n
        self.Kdim = Kdim
        self.Q = Q
        self.x = x
        self.y = y
        self.q = q
        self.a = a
        self.b = b
        self.d = createDistanceMatrix(x, y)


        #### dynamic info for building RMP, PP


        ## update by addRoutetoMaster
        self.A = None ## like self.patterns as before
        self.c = None   # routes costs

        self.routes = None ## route list storing current routes
        

        #### dynamic info (info needed to for state + reward), get from CG iterations from solving current RMP and PP:  
        self.objVal_history = []
        self.total_steps = 0

        ## action with their reduced cost (stored as tuple) ([all the patterns],[(data for those routes)])
        self.available_action = ()

   

        self.count_convergence = 0

        '''  
        Info for column and constraint node features, stored using list,length will change
            column 
                      number of constraint participation
                      current solution value (if not in the basis, 0 -> int or not)
                      columnIsNew

                      column incompatibility degree --> check this

             constraint : shadow price
                          number of columns contributing to the constraint
        '''  
        ## for all the columns (size change)

        # self.RC = [] #### don't quite know how rc works here, so what i will do is that i will only have rc value for actions, all other rc is 0 ? 

        self.In_Cons_Num = []
        self.ColumnSol_Val = []
        self.ColumnIs_Basic = []
        
        ## for all the variable that are in the basis, count the number of times it's in basis, otherwise 0
        self.stay_in = []
        self.stay_out = []
        ## 1-> just left the basis in last iteration, 0 not just left
        self.just_left = []
        self.just_enter = []

        ## 1-> is action node, 0 -> .. useless as we can do this at get_aug_state
        # self.action_node = []



        ## for all the constraints (size fixed)
        self.Shadow_Price = None
        self.In_Cols_Num = []



    def generate_initial_patterns(self):

        impactSol = initializePathsWithImpact(self.d, self.n, self.a, self.b, self.q, self.Q)
        initial_routes = impactSol[:]
        self.routes = deepcopy(initial_routes)

        #print("Impact solution:", routes)

        # impactCost = sum([computeRouteCost(self.routes, self.d) for route in self.routes])


        A = np.zeros((self.n, len(self.routes)))
        c = np.zeros(len(self.routes))  
        addRoutesToMaster(self.routes, A, c, self.d)

        self.A = deepcopy(A)
        self.c = deepcopy(c)
        



    



    # get the constraint participation for each col node and col participation for each cons node
    ## use current patterns to count the non-zeros in the pattern matrix

    ### use A matrix -- read carefully how master and pp is built
    def update_col_con_number(self):
        A = deepcopy(self.A)
        self.In_Cons_Num = np.count_nonzero(A, axis=0)
        self.In_Cols_Num = np.count_nonzero(A, axis=1)
    



    def createMasterProblem(self):
        A = self.A
        costs = self.c
        n = self.n
        vehicleNumber = self.Kdim

        model = gp.Model("Master problem")
        model.Params.OutputFlag = 0
        y = model.addMVar(shape=A.shape[1], vtype=gp.GRB.CONTINUOUS, name="y")
        model.setObjective(costs @ y, gp.GRB.MINIMIZE)
        # Constraints
        model.addConstr(A @ y == np.ones(A.shape[0]))
        model.write("MasterModel.lp")
        model.Params.LogToConsole = 0

        return model


    # def solve_subproblem_return_actions(self, duals):
    # newRoutes = subProblem(n, q, d, a, b, rc, Q)

    def solve_subproblem_return_actions(self, rc, test=False):
        n = self.n
        q = self.q
        d = self.d
        readyt = self.a
        duedate = self.b
        Q = self.Q

        M = gp.GRB.INFINITY     # 1e+100
        # Time windows reduction
        a,b = reduceTimeWindows(n, d, readyt, duedate)
        # Reduce max capacity to boost algorithm
        if sum(q) < Q:
            Q = sum(q)
        T = max(b)

        # Init necessary data structure
        f = list()  # paths cost data struct
        p = list()  # paths predecessor data struct
        f_tk = list()     # cost of the best path that does not pass for
                          # predecessor (we'll call it alternative path)
        paths = []
        paths_tk = []
        for j in range(n+2):
            paths.append([])
            paths_tk.append([])
            for qt in range(Q-q[j]):
                paths[-1].append([])
                paths_tk[-1].append([])
                for tm in range(b[j]-a[j]):
                    paths[-1][-1].append([])
                    paths_tk[-1][-1].append([])
            mat = np.zeros((Q-q[j], b[j] - a[j]))
            p.append(mat - 1)
            f.append(mat + M)
            f_tk.append(mat + M)
        f[0][0,0] = 0
        f_tk[0][0,0] = 0
        L = set()   # Node to explore
        L.add(0)

        # Algorithm
        computation_counter = 0
        while L:
            if test: ## we will test on large instances, so set up some computation limits
                # print("it is test phase! ", test)
                if computation_counter>=20:
                    break
                computation_counter +=1
                    
            else:
                computation_counter +=1 ## but for trainninng we want to train without such limits as instances are small
            

            i = L.pop()
            if i == n+1:
                continue

            # Explore all possible arcs (i,j)
            for j in range(1,n+2):
                if i == j:
                    continue
                for q_tk in range(q[i], Q-q[j]):
                    for t_tk in range(a[i], b[i]):
                        if p[i][q_tk-q[i], t_tk-a[i]] != j:
                            if f[i][q_tk-q[i], t_tk-a[i]] < M:
                                for t in range(max([a[j], int(t_tk+d[i,j])]),\
                                                    b[j]):
                                    if f[j][q_tk, t-a[j]]> \
                                      f[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]:
                                        # if the current best path is suitable to
                                        # become the alternative path
                                        if p[j][q_tk, t-a[j]] != i \
                                          and p[j][q_tk, t-a[j]] != -1 \
                                          and f[j][q_tk, t-a[j]] < M \
                                          and f[j][q_tk,t-a[j]]<f_tk[j][q_tk,t-a[j]]:
                                            f_tk[j][q_tk,t-a[j]] = f[j][q_tk,t-a[j]]
                                            paths_tk[j][q_tk][t-a[j]] = \
                                                    paths[j][q_tk][t-a[j]][:]
                                        # update f
                                        f[j][q_tk,t-a[j]] = \
                                                f[i][q_tk-q[i],t_tk-a[i]] + rc[i,j]
                                        # update path that leads to node j
                                        paths[j][q_tk][t-a[j]] = \
                                                paths[i][q_tk-q[i]][t_tk-a[i]] + [j]
                                        # Update predecessor
                                        p[j][q_tk, t-a[j]] = i
                                        L.add(j)
                                    # if the path is suitable to be the alternative
                                    elif p[j][q_tk, t-a[j]] != i \
                                        and p[j][q_tk, t-a[j]] != -1 \
                                        and f_tk[j][q_tk, t-a[j]] > \
                                                f[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]:
                                        f_tk[j][q_tk,t-a[j]] = \
                                                f[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]
                                        paths_tk[j][q_tk][t-a[j]] = \
                                                paths[i][q_tk-q[i]][t_tk-a[i]]+[j]
                        else:       # if predecessor of i is j
                            if f_tk[i][q_tk-q[i], t_tk-a[i]] < M:
                                for t in range(max([a[j],int(t_tk+d[i,j])]), \
                                                    b[j]):
                                    if f[j][q_tk,t-a[j]] > \
                                            f_tk[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]:
                                        # if the current best path is suitable to
                                        # become the alternative path
                                        if p[j][q_tk, t-a[j]] != i \
                                            and p[j][q_tk, t-a[j]] != -1 \
                                            and f[j][q_tk, t-a[j]] < M \
                                            and f[j][q_tk,t-a[j]] < \
                                                    f_tk[j][q_tk,t-a[j]]:
                                            f_tk[j][q_tk,t-a[j]] = f[j][q_tk,t-a[j]]
                                            paths_tk[j][q_tk][t-a[j]] = \
                                                    paths[j][q_tk][t-a[j]][:]
                                        # update f, path and bucket
                                        f[j][q_tk,t-a[j]] = \
                                            f_tk[i][q_tk-q[i],t_tk-a[i]] + rc[i,j]
                                        paths[j][q_tk][t-a[j]] = \
                                            paths_tk[i][q_tk-q[i]][t_tk-a[i]] + [j]
                                        p[j][q_tk,t-a[j]] = i
                                        L.add(j)
                                    # if the alternative path of i is suitable to
                                    # be the alternate of j
                                    elif p[j][q_tk, t-a[j]] != i \
                                        and p[j][q_tk, t-a[j]] != -1 \
                                        and f_tk[j][q_tk,t-a[j]] > \
                                                f_tk[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]:
                                        f_tk[j][q_tk, t-a[j]] = \
                                            f_tk[i][q_tk-q[i],t_tk-a[i]]+rc[i,j]
                                        paths_tk[j][q_tk][t-a[j]] = \
                                            paths_tk[i][q_tk-q[i]][t_tk-a[i]] + [j]

        # Return all the routes with negative cost
        routes = list()
        rcosts = list()
        qBest, tBest = np.where(f[n+1] < -1e-9)

        for i in range(len(qBest)):
            newRoute = [0] + paths[n+1][qBest[i]][tBest[i]]
            if not newRoute in routes:
                routes.append(newRoute)
                rcosts.append(f[n+1][qBest[i]][tBest[i]])

        # print("New routes:", routes, flush=True)


        # print("reduced cost?", rcosts)


        costs = np.zeros(len(routes))
        for i in range(len(routes)):
            cost = d[routes[i][0],routes[i][1]]
            for j in range(1,len(routes[i])-1):
                cost += d[routes[i][j], routes[i][j+1]]
            costs[i] = cost

        costs = list(costs)
        # print("new routes",routes)


        return routes,rcosts,costs  ## return all avaliable actions (new routes generated, called routes here) and their rc

        

    def initialize(self,test_or_not=False):

        self.generate_initial_patterns()
        self.total_steps = 0
        routes = self.routes

        self.update_col_con_number()

        master_problem = self.createMasterProblem()
        master_problem.optimize()

        # # Compute reduced costs
        # constr = masterModel.getConstrs()
        # pi_i = [0.] + [const.pi for const in constr] + [0.]
        # for i in range(n+2):
        #     for j in range(n+2):
        #         rc[i,j] = d[i,j] - pi_i[i]

        # if not np.where(rc < -1e-9):
        #     break

        # self.RC = np.zeros(len(self.routes))


        self.ColumnSol_Val = np.asarray(master_problem.x)
        self.ColumnIs_Basic = np.asarray(master_problem.vbasis)+np.ones(len(routes))
        self.objVal_history.append(master_problem.objVal)


        dual_variables = [constraint.pi for constraint in master_problem.getConstrs()]
        self.Shadow_Price = dual_variables

        # Compute reduced costs
        constr = master_problem.getConstrs()
        pi_i = [0.] + [const.pi for const in constr] + [0.]

        rc = np.zeros((self.n+2,self.n+2))
        for i in range(self.n+2):
            for j in range(self.n+2):
                rc[i,j] = self.d[i,j] - pi_i[i]
        # print("reduced cost",rc)


        too_long = False
        time_before = time.time()
        columns_to_select,reduced_costs,route_costs = self.solve_subproblem_return_actions(rc,test_or_not)
        time_after = time.time()
        test=test_or_not
        if not test:
            if time_after - time_before >=100.0: ## if the instance takes too long to solve during training, skip it
                print("initializing time :  ", time_after -time_before)
                too_long = True
        else:
            too_long = False ## test every instances

        self.available_action = (columns_to_select,[reduced_costs,route_costs])


        self.stay_in = list(np.zeros(len(routes)))
        self.stay_out = list(np.zeros(len(routes)))
        self.just_left = list(np.zeros(len(routes)))
        self.just_enter = list(np.zeros(len(routes)))

        reward = 0
        is_done = False
  
        return reward, is_done, too_long
        

    def step(self,action,test_or_not=False):
        self.total_steps +=1
        is_done = False

        ## historical info
        last_columns_to_select, columns_info = deepcopy(self.available_action) 
        last_basis= deepcopy(self.ColumnIs_Basic[:])
        last_basis = np.append(last_basis,0)


        self.routes.append(action)
        idx = 0
        for one_act in last_columns_to_select:
            if one_act == action:
                break
            idx+=1
        
        routes = deepcopy(self.routes)

        ## just append one cost
        self.c = np.append(self.c,columns_info[1][idx])

        ########################################
        # self.rc.appennd(columns_info[0][idx])
        ########################################

        



        
        # c = np.zeros(len(self.routes))  
        # addRoutesToMaster(self.routes, A, c, self.d)



        add_A = np.zeros((self.n, 1))

        for j in range(1,len(action)-1):
            add_A[action[j]-1] += 1

        # print(add_A)
        self.A = np.c_[self.A, add_A]

        self.update_col_con_number()
        before_master_problem = time.time()
        master_problem = self.createMasterProblem()
        master_problem.optimize()
        # print("solving master problem takes : ", time.time()-before_master_problem, " seconds")

        dual_variables = [constraint.pi for constraint in master_problem.getConstrs()]
        self.Shadow_Price = dual_variables

        self.ColumnSol_Val = np.asarray(master_problem.x)
        self.ColumnIs_Basic = np.asarray(master_problem.vbasis)+np.ones(len(routes))

        #### you can either stay in the basis, leave the basis, or enter the basis
        difference = last_basis - self.ColumnIs_Basic 

        ### update the dynamic basis info based on difference
        self.just_left = list(np.zeros(len(difference)-1))
        self.just_enter = list(np.zeros(len(difference)-1))
        for i in range(len(difference)-1):
            if difference[i] == 1:
                
                self.just_left[i] = 1
                self.stay_in[i] = 0
            elif difference[i] == -1:
                
                self.just_enter[i] = 1
                self.stay_out[i] = 0
            elif difference[i] == 0:
                if last_basis[i] == 1:
                    self.stay_in[i]+=1
                else:
                    self.stay_out[i]+=1

        # append info for the new node; for just enter, look at column is basic
        self.just_left.append(0)
        self.stay_out.append(0)
        self.stay_in.append(0)

        if self.ColumnIs_Basic[-1] == 1:
            self.just_enter.append(1)
        else:
            self.just_enter.append(0)
   
        self.objVal_history.append(master_problem.objVal)



        rc = np.zeros((self.n+2,self.n+2))

        # Compute (rc, don't know what it is) used for building subproblem
        constr = master_problem.getConstrs()
        pi_i = [0.] + [const.pi for const in constr] + [0.]
        for i in range(self.n+2):
            for j in range(self.n+2):
                rc[i,j] = self.d[i,j] - pi_i[i]
        # print("reduced cost",rc)


        new_routes,action_rcosts,action_costs = self.solve_subproblem_return_actions(rc,test_or_not)
        if new_routes == []:
            is_done = True
            reward = 0.5*(self.objVal_history[-2] - self.objVal_history[-1])/self.objVal_history[0]+100

        else:
            self.available_action = (new_routes,[action_rcosts,action_costs])
            reward = 0.5*(self.objVal_history[-2] - self.objVal_history[-1])/self.objVal_history[0] ## normalization term
            reward -= 10



        return reward, is_done