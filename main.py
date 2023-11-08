from Config import Config
from AuxilaryFunctions2 import generate_and_save_data
import os
import shutil

import multiprocessing
from multiprocessing import  Pool,cpu_count
import numpy as np
import time

# homepath = 'data/'
homepath = Config.dataDirectory
# homepath += '/'
#
if not os.path.exists(homepath[:-1]):
   os.makedirs(homepath[:-1])
else:
   # reset it
   shutil.rmtree(homepath)
   os.makedirs(homepath[:-1])

folders = ['images','labels']
subFolders = ['train','val']
for folder in folders:
   subPath = homepath + folder
   if not os.path.exists(subPath):
       os.makedirs(subPath)
   for subFolder in subFolders:
       subSubPath = subPath + '/' + subFolder
       if not os.path.exists(subSubPath):
           os.makedirs(subSubPath)


generate_and_save_data(22)

# def init_pool_process():
#     np.random.seed()
#
# if __name__ == '__main__':
#     # multiprocessing case
#     print('Process Starting')
#     startTime = time.time()
#     amount = Config.amountOfData
#     pool_obj = multiprocessing.Pool(initializer=init_pool_process)
#     pool_obj.map(generate_and_save_data, range(0,amount))
#     pool_obj.close()
#     endTime = time.time()
#
#     print('Finish Running')
#     print('Average Time: ' + str((endTime - startTime)/amount))




