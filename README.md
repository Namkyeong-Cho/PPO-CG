# PPO-CG

This repository is an official implementation of "Policy optimization in reinforcement learning for column generation" 


## Prerequisites
```bash
git clone LINKE
cd PPO-CG
pip install -r requirements.txt
```
### For each task 
### To train run the following codes\
```bash
cd CSP/VRPTW (depending on the task)
python mainy.py --RL_algorithm {algorithm_choice} --train
```
algorithm_choice = DQN or PPO


 Once training is done, the model will be saved in output files.To train test the following codes
 ### To test the model run the following codes
```bash
cd CSP/VRPTW (depending on the task)
python mainy.py --RL_algorithm {algorithm_choice} --test
```
algorithm_choice = DQN or PPO or no_RL

#### Output file will be saved in ./task_name/outputs/

