from re import L
import environment
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_ACTION_DAY, KEY_ACTION_NUM, KEY_ACTION_VALUE, KEY_DATA_TARGET_NAME

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time

import numpy as np
import pandas as pd

# hyperparamters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 300
max_test_ep = 400

num_target = '000810'

class ActorCritic(nn.Module):
  def __init__(self):
    super(ActorCritic, self).__init__()
    self.fc1 = nn.Linear(336, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 32)
    self.fc_pi = nn.Linear(32, 1)
    self.fc_v = nn.Linear(32, 1)
  
  def pi(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    pi = F.sigmoid(self.fc_pi(x))
    return pi
  
  def v(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    v = self.fc_v(x)
    return v

def train(global_model, rank):
  local_model = ActorCritic()
  local_model.load_state_dict(global_model.state_dict())

  optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

  env = environment.Environment()

  for n_epi in range(max_train_ep):
    done = False
    s = env.reset()
    while not done:
      s_list, a_list, r_list = [], [], []
      for t in range(update_interval):
        df_target = s[s[KEY_DATA_TARGET_NUM] == num_target]
        df_target = df_target.drop([KEY_DATA_TARGET_NUM], axis=1)
        inputs = np.array(list(df_target.values)[0])
        day_target = list(df_target.index)[0]
        prob = local_model.pi(torch.from_numpy(inputs).float())
        a = prob.item()
        action = [{KEY_ACTION_DAY: day_target, KEY_ACTION_NUM: num_target, KEY_ACTION_VALUE: prob.item()}]
        s_prime, r, done = env.step(action)

        s_list.append(inputs)
        a_list.append(a)
        r_list.append(r)

        s = s_prime
        if done:
          break
      df_target = s[s[KEY_DATA_TARGET_NUM] == num_target]
      df_target = df_target.drop([KEY_DATA_TARGET_NUM], axis=1)
      inputs = np.array(list(df_target.values)[0])
      s_final = torch.tensor(torch.from_numpy(inputs), dtype=torch.float)
      R = 0.0 if done else local_model.v(s_final).item()
      td_target_list = []
      for reward in r_list[::-1]:
        R = gamma * R + reward
        td_target_list.append([R])
      td_target_list.reverse()

      s_batch, a_batch, td_target = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), torch.tensor(td_target_list)
      advantage = td_target - local_model.v(s_batch)

      pi = local_model.pi(s_batch)
      # pi_a = pi.gather(1, a_batch)
      loss = -(pi * advantage.detach()) + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())
      optimizer.zero_grad()
      loss.mean().backward()
      for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        global_param._grad = local_param.grad
      optimizer.step()
      local_model.load_state_dict(global_model.state_dict())
  print('Training process {} reached maximum episode.'.format(rank))

def test(global_model):
  env = environment.Environment()
  scores = []
  print_interval = 5

  for n_epi in range(max_test_ep):
    done = False
    s = env.reset()    
    while not done:
      df_target = s[s[KEY_DATA_TARGET_NUM] == num_target]
      df_target = df_target.drop([KEY_DATA_TARGET_NUM], axis=1)
      inputs = np.array(list(df_target.values)[0])
      day_target = list(df_target.index)[0]
      prob = global_model.pi(torch.from_numpy(inputs).float())
      a = prob.item()
      action = [{KEY_ACTION_DAY: day_target, KEY_ACTION_NUM: num_target, KEY_ACTION_VALUE: prob.item()}]
      s_prime, r, done = env.step(action)
      if n_epi % print_interval == 0 and n_epi != 0:
        print(day_target, ': ', action, ', Reward: ', r)
      scores.append(r)
      s = s_prime
    
    if n_epi % print_interval == 0 and n_epi != 0:
      print("# of episode :{}, avg score : {:.1f}".format(n_epi, np.mean(scores)))
      scores = []

if __name__ == '__main__':
  global_model = ActorCritic()
  global_model.share_memory()

  processes = []
  for rank in range(n_train_processes + 1):
    if rank == 0:
      p = mp.Process(target=test, args=(global_model, ))
    else:
      p = mp.Process(target=train, args=(global_model, rank,))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()