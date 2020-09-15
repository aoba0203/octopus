from operator import mod
import environment
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_ACTION_DAY, KEY_ACTION_NUM, KEY_ACTION_VALUE, KEY_DATA_TARGET_NAME
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

num_target = '000810'
LOSS = 0.0
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class ReplayBuffer():
  def __init__(self):
    self.buffer = collections.deque(maxlen=buffer_limit)
  
  def put(self, transition):
    self.buffer.append(transition)
  
  def sample(self, n):
    mini_batch = random.sample(self.buffer, n)
    s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

    for transition in mini_batch:
      s, a, r, s_prime, done_mask = transition
      s_list.append(s)
      a_list.append([a])
      r_list.append([r])
      s_prime_list.append(s_prime)
      done_mask_list.append([done_mask])
    
    return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
            torch.tensor(r_list), torch.tensor(s_prime_list, dtype=torch.float), \
            torch.tensor(done_mask_list)
  def size(self):
    return len(self.buffer)
  
class Qnet(nn.Module):
  def __init__(self):
    super(Qnet, self).__init__()
    self.fc1 = nn.Linear(336, 512)    
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 64)
    self.fc4 = nn.Linear(64, 8)
    self.fc5 = nn.Linear(8, 2)

  def forward(self, _x):
    x = F.relu(self.fc1(_x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return x

  def sampleAction(self, obs, epsilon):
    out = self.forward(obs)
    coin = random.random()
    if coin < epsilon:
      return random.randint(0, 1)
    else:
      return out.argmax().item()

def train(q, q_target, memory, optimizer):
  for i in range(10):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)
    LOSS = loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def makeInput(state):
  df_target = state[state[KEY_DATA_TARGET_NUM] == num_target]
  df_target = df_target.drop([KEY_DATA_TARGET_NUM], axis=1)
  if len(list(df_target.values)) == 0:    
    return None
  inputs = np.array(list(df_target.values)[0])
  inputs = torch.from_numpy(inputs).float().to(device)
  day_target = list(df_target.index)[0]
  return day_target, inputs

def main():
  env = environment.Environment()
  q = Qnet().to(device)
  q_target = Qnet().to(device)
  q_target.load_state_dict(q.state_dict)
  memory = ReplayBuffer()

  print_interval = 20
  score = 0.0
  optimizer = optim.Adam(q.parameters(), lr=learning_rate)

  for n_epi in range(10000):
    epsilon =  max(0.01, 0.08 - 0.01 * (n_epi/200))
    state = env.reset()
    done = False
    actions = []
    while not done:
      day_target, inputs = makeInput(state)
      a = q.sampleAction(inputs, epsilon)
      actions.append({KEY_ACTION_DAY: day_target, KEY_ACTION_NUM: num_target, KEY_ACTION_VALUE: a.item()})
      next_state, reward, modified_r, done = env.step(actions)
      done_mask = 0.0 if done else 1.0
      memory.put((state, a, reward/100.0, next_state, done_mask))

      score += modified_r
      if done:
        break
    if memory.size() > 2000:
      train(q, q_target, memory, optimizer)
    
    if n_epi % print_interval == 0 and n_epi != 0:
      print('# of episode: {}, avg score: {}, loss: {}'.format(n_epi, round(score/ print_interval, 3), LOSS))
      score = 0
      


    
