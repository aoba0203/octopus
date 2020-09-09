from utils import definitions
from bs4 import BeautifulSoup
import urllib
from urllib import parse, request
import requests
import pandas as pd
import os
from multiprocessing import Pool
import threading

class TargetInfoGather:
  def __init__(self, _list_target):
    self.path_raw = definitions.getDataRawPath()
    self.list_target = _list_target
  
  def __get_last_page_num_demand(self, _stock_num):
    content = requests.get(('http://finance.naver.com/item/frgn.nhn?code=' + _stock_num)).text
    parsed_text = BeautifulSoup(content, features='lxml')
    pgrrs = parsed_text.findAll('td', {'class': 'pgRR'})
    if len(pgrrs) == 0:
        return 0
    parsed = parse.urlparse(pgrrs[0].a['href'])
    parsed_qs = parse.parse_qs(parsed.query, keep_blank_values=True)
    return int(parsed_qs['page'][0])

  def __get_parse_target_info(self, _stock_num, _page_num):
    # parse index- end | diff | open | high | low | volume
    parse_url = 'http://finance.naver.com/item/sise_day.nhn?code=' + _stock_num + '&page=' + _page_num
    soup = BeautifulSoup(request.urlopen(parse_url), 'lxml')
    elements_day = soup.findAll('td', {'align': 'center'})
    elements_num = soup.findAll('td', {'class': 'num'})

    count = 0
    diff = 1
    parsed_infos = []
    for elements in elements_day:
      info = []
      day = elements.text
      info.append(day)
      for idx in range(6):
        idx += (count * 6)
        if len(elements_num[idx].text) == 1:
          continue
        parse_value = int(elements_num[idx].text.replace(',', ''))
        if idx % 6 == 1:
          if parse_value == 0:
            diff = 0
          elif 'ico_down.gif' in elements_num[idx].img['src']:
            # elif elements_num[idx].img['src'] == 'http://imgstock.naver.com/images/images4/ico_down.gif':
            diff = -1
        parsed_info = diff * int(elements_num[idx].text.replace(',', ''))
        info.append(parsed_info)
        diff = 1
      count += 1
      parsed_infos.append(info)
    return parsed_infos

  def __parse_target_demand(self, _stock_num, _page_num):
    target_demand = []
    # parse index- day | agency | foreign
    parse_url = 'http://finance.naver.com/item/frgn.nhn?code=' + _stock_num + '&page=' + _page_num
    soup = BeautifulSoup(request.urlopen(parse_url), 'lxml')
    elements = soup.findAll('tr', {'onmouseover': 'mouseOver(this)'})

    for element in elements:
      demand = []
      text_array = element.text.split()
      for idx in range(len(text_array)):
        if idx % 9 == 0: demand.append(text_array[idx])
        if idx % 9 == 5: demand.append(float(text_array[idx].replace(',', '')))
        if idx % 9 == 6: demand.append(float(text_array[idx].replace(',', '')))
      target_demand.append(demand)
    return target_demand
  
  def __write_target_info_from_web(self, _stock_num, _last_page_num):
    target_info = []
    target_demand = []
#     last_page_num = __get_last_page_num_demand(stock_num)
    for page_num in range(1, (_last_page_num * 2)):
      print(str(page_num) + '/' + str((_last_page_num * 2)))
      parsed_info = self.__get_parse_target_info(_stock_num, str(page_num))
      target_info.extend(parsed_info)
    for page_num in range(1, _last_page_num):
      print(str(page_num) + '/' + str(_last_page_num))
      parsed_demand = self.__parse_target_demand(_stock_num, str(page_num))
      target_demand.extend(parsed_demand)    
    df_info = pd.DataFrame(target_info)
    df_demand = pd.DataFrame(target_demand)
    df_info.set_index(0, inplace=True)
    df_demand.set_index(0, inplace=True)
    df_info = df_info.dropna()
    df_demand = df_demand.dropna()
    # print(stock_num, ': ', df_info.shape, ', ', df_demand.shape)
    if df_info.shape[0] > df_demand.shape[0]:
        df_info = df_info[:df_demand.shape[0]]
    elif df_info.shape[0] < df_demand.shape[0]:
        df_demand = df_demand[:df_info.shape[0]]
    df = pd.concat([df_info, df_demand], axis=1)
    df.sort_index(inplace=True, ascending=False)
    df = df.dropna()
    df.columns = ['close', 'diff', 'open', 'high', 'low', 'volume', 'agency', 'foreign']
#     df = __add_stock_fundamental(df, stock_num)
    path_csv_file = os.path.join(self.path_raw, _stock_num + '.csv')
    df.to_csv(path_csv_file)
  
  def update_stock_info(self, _stock_num):    
    path_csv_file = os.path.join(self.path_raw, _stock_num + '.csv')
    target_info = []
    target_demand = []
    df_csv = pd.read_csv(path_csv_file, index_col=0)
    target_info.extend(self.__get_parse_target_info(_stock_num, '1'))
    target_info.extend(self.__get_parse_target_info(_stock_num, '2'))
    target_demand.extend(self.__parse_target_demand(_stock_num, '1'))
    df_info = pd.DataFrame(target_info)
    df_demand = pd.DataFrame(target_demand)
    df_info.set_index(0, inplace=True)
    df_demand.set_index(0, inplace=True)
    df_update = (pd.concat([df_info, df_demand], axis=1)).sort_index(ascending=False)
    df_update.columns = ['close', 'diff', 'open', 'high', 'low', 'volume', 'agency', 'foreign']
    df_result = df_update.combine_first(df_csv)
    df_result.sort_index(inplace=True, ascending=False)
    # path_csv_file = os.path.join(self.path_raw, _stock_num + '.csv')
    df_result.to_csv(path_csv_file)
  
  def __save_update_target_info(self, _stock_num):
    print('__save_update_target_info')
    stock_num = _stock_num.rjust(6, '0')    
    try:
      if not os.path.exists(os.path.join(self.path_raw, _stock_num + '.csv')):
        last_page_num = self.__get_last_page_num_demand(stock_num)
        if last_page_num < 30:
          print(stock_num + ': is too short')
          return
        self.__write_target_info_from_web(stock_num, last_page_num)
      else:
        # self.__update_stock_info(stock_num)
        print('alread exist file: ', _stock_num)
    except Exception as e:
        print('Except TargetInfoGather - __save_update_target_info: ', stock_num)
        print(e)
    return

  def save_and_updates_target_info(self):
    print('save_and_updates_target_info')
    pool = Pool()
    for target_num in self.list_target:
    #   print(target_num)
    #   t = threading.Thread(target=self.__save_update_target_info, args=(target_num,))
    #   t.start()
      self.__save_update_target_info(target_num)
    # for i, _ in enumerate(pool.map(self.__save_update_target_info, self.list_target)):
    #     pass
    