import pandas as pd
import os
from . import target_info_gather, target_info_process, target_list_manager
import glob
from multiprocessing import Pool
from utils import definitions
from utils.definitions import KEY_DATA_TARGET_NUM, KEY_DATA_TARGET_NAME

class DataManager:
  def __init__(self):
    self.target_list = target_list_manager.TargetManager()
    self.target_num_list = self.target_list.getTargetList()
    self.info_gather = target_info_gather.TargetInfoGather(self.target_num_list)
    self.info_process = target_info_process.TargetInfoProcess()
  
  def getDataset(self):
    dir_processed = definitions.getDataProcessedPath()
    path_processed = os.path.join(dir_processed, '*.csv')
    list_csv_file = glob.glob(path_processed)
    list_df = []
    for csv_file in list_csv_file:
      target_num = os.path.splitext(os.path.basename(csv_file))[0]
      df_from_file = pd.read_csv(csv_file, index_col=0)
      df_from_file[KEY_DATA_TARGET_NUM] = target_num
      df_from_file[KEY_DATA_TARGET_NAME] = df_from_file['0_candle_size'].shift()
      df_from_file = df_from_file.dropna()
      list_df.append(df_from_file)
    df = pd.concat(list_df, join='outer')
    df['num_convert'] = pd.factorize(df['num'])[0]
    return df
  
  def makeDataset(self):    
    print('START - save_and_updates_target_info')
    self.info_gather.save_and_updates_target_info()
    print('START - write_extra_info_by_list')
    self.info_process.write_extra_info_by_list(self.target_num_list)
    

    