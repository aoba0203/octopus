import pandas as pd
from utils import definitions
import os
import glob

TARGET_FILE_NAME = 'stock_kosp20.csv'

class TargetManager:
  def __init__(self):
    dir_target = definitions.getDataTargetsPath()
    path_target = os.path.join(dir_target, TARGET_FILE_NAME)
    self.df_target = pd.read_csv(path_target, names=['num','name'], dtype={'num':str})
  
  def getTargetList(self):
    return list(self.df_target['num'].values)

  def getTargetName(self, _target_num):
    return self.df_target[self.df_target.num == _target_num].name.values[0]

  def getTargetNum(self, _target_name):
    return self.df_target[self.df_target.name == _target_name].num.values[0]
  