#%%
import os
import pandas as pd
from utils import definitions
from data import target_list_manager, target_info_gather, target_info_process

print(definitions.getProjectRootPath())
target = target_list_manager.TargetManager()

#%%
process = target_info_process.TargetInfoProcess()
process.write_extra_infos('005930')
# %%

gather = target_info_gather.TargetInfoGather(['005930'])
# %%
gather.save_and_updates_target_info()
# %%
target.df_target
# %%

pd.read_csv('data\\processed\\005930.csv')

# %%
