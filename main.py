#%%
import os
import pandas as pd
from utils import definitions
from data import target_list_manager, target_info_gather, target_info_process

print(definitions.getProjectRootPath())
target = target_list_manager.TargetManager()
#%%
list(target.df_target['num'].values)
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

import glob
csv_path = os.path.join(definitions.getDataRawPath(), '*.csv')
print(csv_path)
glob.glob(csv_path)

# %%
