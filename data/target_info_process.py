import os
import pandas as pd
import numpy as np
from utils import definitions
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool

class TargetInfoProcess:
  def __init__(self):
    self.path_raw = definitions.getDataRawPath()
    self.path_processed = definitions.getDataProcessedPath()
  
  def __set_extra_info(self, _df):
    _df['candle_size'] = (((_df['close'] - _df['open']) / _df['open']) * 100.0)
    _df['percent'] = (_df['diff'] / _df['close'].shift(-1)) * 100
    _df['volume'] = (_df['volume'] * _df['close'])
    _df['target'] = _df['candle_size'].shift()
    return _df
  
  def __set_scaled_volume(self, _df, _targets=['close', 'volume', 'agency', 'foreign']):
    mms = MinMaxScaler(feature_range=(-10, 10))
    _df[_targets] = mms.fit_transform(_df[_targets])
  
  def __set_scaled_price(self, _df, _targets=['open', 'high', 'low']):
    _df['diff'] = (_df['diff'] / _df['close']) * 100
    for target in _targets:
        _df[target] = ((_df[target] - _df['close']) / _df['close']) * 100
  
  def __set_mv_grad(self, _df):
    for duration in [3, 5, 10, 20, 60, 120, 200]:
      _df['mv_value' + str(duration)] = _df['close'].rolling(duration).mean().shift(-(duration - 1))
      _df['mv_value' + str(duration) + '_grad'] = np.flip(np.gradient(_df['mv_value' + str(duration)].sort_index().values), 0)
      _df['mv_volume' + str(duration)] = _df['volume'].rolling(duration).mean().shift(-(duration - 1))
      _df['mv_volume' + str(duration) + '_grad'] = np.flip(np.gradient(_df['mv_volume' + str(duration)].sort_index().values), 0)

      _df['mv_foreign' + str(duration)] = _df['foreign'].rolling(duration).mean().shift(-(duration - 1))
      _df['mv_foreign' + str(duration) + '_grad'] = np.flip(np.gradient(_df['mv_foreign' + str(duration)].sort_index().values), 0)
      _df['mv_agency' + str(duration)] = _df['agency'].rolling(duration).mean().shift(-(duration - 1))
      _df['mv_agency' + str(duration) + '_grad'] = np.flip(np.gradient(_df['mv_agency' + str(duration)].sort_index().values), 0)

  def __get_stacked_dataset(self, _df, _stack_size=5):
    data_list = []
    columns = []
    for idx in range(len(_df) - _stack_size):
      data_list.append(_df.iloc[idx:idx + _stack_size].values.flatten())
    for i in range(_stack_size):
      for column in _df.columns:
        columns.append(str(i) + '_' + column)
    return pd.DataFrame(data_list, columns=columns)

  def write_extra_infos(self, _stock_num):
    df_day_list =[]
    try:
      file_path = os.path.join(self.path_raw, _stock_num + '.csv')
      if not os.path.exists(file_path):
        print('file not exists')
        return
      df_ori = pd.read_csv(file_path, index_col=0)
      df_ori = df_ori.dropna()    
      self.__set_extra_info(df_ori)
      self.__set_scaled_price(df_ori)
      self.__set_scaled_volume(df_ori)        

      for idx in range(1, len(df_ori) - 204):        
        df_s = df_ori[idx: idx + 205]
        self.__set_mv_grad(df_s)    
        df_indexs = df_s.index.copy()
        df_s = df_s.reset_index()    
        df_s = df_s.drop(df_s.columns[0], axis=1)
        df_s = self.__get_stacked_dataset(df_s)
        df_s.index = df_indexs[:len(df_s)]
        df_s = df_s.dropna()
        df_day_list.append(df_s)
      if len(df_day_list) == 0:
        print('len(df_day_list) == 0')
        return
      df_result = pd.concat(df_day_list)
      df_result = df_result.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
      df_result.to_csv(os.path.join(self.path_processed, _stock_num + '.csv'))
    except Exception as e:
      print('Except: __new_write_seperate_day_dataframe = ', _stock_num)
      print(e)

  def write_extra_info_by_list(self, _target_list):
    cpus = definitions.getNumberOfCore()
    with Pool(cpus) as p:
      p.map(self.write_extra_infos, _target_list)
    # for target_num in _target_list:
    #   self.write_extra_infos(target_num)
    