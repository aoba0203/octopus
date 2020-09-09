from data import data_manager as dtm
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_ACTION_DAY, KEY_ACTION_NUM, KEY_ACTION_VALUE
import random

class Env:
  def __init__(self, _epoch=5):
    self.data_manager = dtm.DataManager()
    self.df = self.data_manager.getDataset()
    self.list_day = self.__getDayList()
    self.train_count = 0
    self.epoch = _epoch
    return
  
  def __getDayList(self):
    list_day = []

    return list_day

  def __isDone(self):
    self.train_count += 1
    return self.train_count > (len(self.list_day) * self.epoch)

  def __getNextState(self):
    len_max_day = len(self.list_day) - 1
    idx_day = random.randint(1, len_max_day)
    day = self.list_day[idx_day]
    df_day = self.df[[day]]
    return df_day

  def __calcReward(self, _list_dic_action):
    reward = 0
    for dic_action for _list_dic_action:
      day = dic_action[KEY_ACTION_DAY]
      num = dic_action[KEY_ACTION_NUM]
      action = dic_action[KEY_ACTION_VALUE]
      if action > 0.5:
        df_day = self.df[[day]]
        reward += df_day[df_day[KEY_DATA_TARGET_NUM] == num]]['target'].values[0]
    return reward

  def reset(self):
    self.train_count = 0
    return
  
  # action = [{'day': day, target_num':num, 'action':0 or 1}]
  def step(self, _list_dic_action):
    next_state = self.__getNextState()
    reward = self.__calcReward(_list_dic_action)
    done = self.__isDone()
    
    return next_state, reward, done