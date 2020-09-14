# https://github.com/seungeunrho/minimalRL
#%%
import os
import pandas as pd
from torch.utils import data
from utils import definitions
from data import data_manager, target_list_manager, target_info_gather, target_info_process

if __name__ == '__main__':
  dtm = data_manager.DataManager()
  dtm.makeDatasetParallel()
# %%
import pandas as pd

# %%
pd.read_csv('data\\processed\\000270.csv')
# %%
