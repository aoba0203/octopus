#%%
import environment
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_ACTION_DAY, KEY_ACTION_NUM, KEY_ACTION_VALUE, KEY_DATA_TARGET_NAME

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

LEARNING_RATE = 0.001
GAMMA = 0.98

num_target = '000810'

class Policy(nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.data = []

    self.fc1 = nn.Linear(336, 256)
    # self.fc2 = nn.Linear(512, 1024)
    # self.fc3 = nn.Linear(1024, 512)
    # self.fc4 = nn.Linear(512, 256)
    self.fc5 = nn.Linear(256, 64)
    self.fc6 = nn.Linear(64, 8)
    self.fc7 = nn.Linear(8, 2)
    self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
  
  def forward(self, _x):
    x = F.relu(self.fc1(_x))
    # x = F.relu(self.fc2(x))
    # x = F.relu(self.fc3(x))
    # x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    x = F.softmax(self.fc7(x))
    return x
  
  def put_data(self, _item):
    self.data.append(_item)
  
  def train_net(self):
    R = 0
    self.optimizer.zero_grad()
    for r, net in self.data[::-1]:
      R = r + (GAMMA * R)
      loss = -torch.log(net) * (R)
      loss.backward()
    self.optimizer.step()
    self.data = []

def main():
  env = environment.Environment()
  policy = Policy()
  policy.cuda()
  score = 0.0
  list_score = []
  print_interval = 1

  for episode in range(10000):
    state = env.reset()
    done = False
    while not done:
      actions = []
      # prob = None
      # a = None
      # for num_target in list(state[KEY_DATA_TARGET_NUM].values):
      df_target = state[state[KEY_DATA_TARGET_NUM] == num_target]
      df_target = df_target.drop([KEY_DATA_TARGET_NUM], axis=1)
      if len(list(df_target.values)) == 0:
        state = env.getNextState()
        continue
      inputs = np.array(list(df_target.values)[0])
      day_target = list(df_target.index)[0]
      prob = policy(torch.from_numpy(inputs).float())
      m = Categorical(prob)
      a = m.sample()
      actions.append({KEY_ACTION_DAY: day_target, KEY_ACTION_NUM: num_target, KEY_ACTION_VALUE: a.item()})
      next_state, reward, modified_r, done = env.step(actions)
      policy.put_data((modified_r, prob[a]))
      list_score.append(reward)
      state = next_state
    policy.train_net()
    if episode % print_interval == 0 and episode != 0:
      print('# of episode: {}, avg score: {}'.format(episode, round(np.sum(list_score)/ print_interval, 3)))
      list_score = []
  env.close()

if __name__ == '__main__':
  main()

