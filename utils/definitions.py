from genericpath import exists
import os
from os import makedirs
from pathlib import Path
from . import utils

def getProjectRootPath():
  file = Path(os.path.abspath('definitions.py'))
  parent= file.parent
  # return os.path.dirname(os.path.abspath(__file__))
  return parent

def getDataPath():
  root_path = Path(getProjectRootPath())
  data_path = os.path.join(root_path, 'data')
  return utils.makeDirs(data_path)

def getDataRawPath():
  data_path = getDataPath()
  raw_path = os.path.join(data_path, 'raw')
  return utils.makeDirs(raw_path)

def getDataTargetsPath():
  data_path = getDataPath()
  raw_path = os.path.join(data_path, 'targets')
  return utils.makeDirs(raw_path)

def getDataProcessedPath():
  data_path = getDataPath()
  processed_path = os.path.join(data_path, 'processed')
  return utils.makeDirs(processed_path)

# def makeDirs(_path):
#   if not os.path.exists(_path):
#     os.makedirs(_path)
#   return _path
