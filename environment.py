from data import data_manager as dtm
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_ACTION_DAY, KEY_ACTION_NUM, KEY_ACTION_VALUE, KEY_DATA_TARGET_NAME
import random

class Environment:
  def __init__(self, _epoch=1):
    self.data_manager = dtm.DataManager()
    self.df = self.data_manager.getDataset()
    self.list_day = self.__getDayList()
    self.train_count = 0
    self.epoch = _epoch
    return
  
  def __getDayList(self):
    list_day = list(self.df.index.unique())
    random.shuffle(list_day)
    return list_day

  def __isDone(self):
    self.train_count += 1
    return self.train_count > (len(self.list_day) * self.epoch)

  def __getNextState(self):
    idx = self.train_count % (len(self.list_day))
    day = self.list_day[idx]
    df_day = self.df.loc[[day]]
    if KEY_DATA_TARGET_NAME in df_day.columns:
      df_day = df_day.drop([KEY_DATA_TARGET_NAME], axis=1)
    return df_day

  def __calcReward(self, _list_dic_action):
    reward = 0
    for dic_action in _list_dic_action:
      day = dic_action[KEY_ACTION_DAY]
      num = dic_action[KEY_ACTION_NUM]
      action = dic_action[KEY_ACTION_VALUE]
      if action > 0.5:
        df_day = self.df.loc[[day]]
        reward += df_day[df_day[KEY_DATA_TARGET_NUM] == num]['target'].values[0]
    return reward

  def reset(self):
    self.train_count = 0
    return self.__getNextState()  
  
  def step(self, _list_dic_action):
    next_state = self.__getNextState()
    reward = self.__calcReward(_list_dic_action)
    done = self.__isDone()
    
    return next_state, reward, done