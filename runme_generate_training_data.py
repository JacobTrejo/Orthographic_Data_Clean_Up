from Programs.Config import Config
from Programs.Aquarium import Aquarium
import os
import shutil
import multiprocessing
import numpy as np
import time

def genData(idx):
    aquarium = Aquarium(idx)
    aquarium.draw()
    aquarium.save_image()
    aquarium.save_annotations()


homepath = Config.dataDirectory

if not os.path.exists(homepath[:-1]):
   os.makedirs(homepath[:-1])
# # Not resting it no more because it is strange, should try looking for a better function
# else:
#    # reset it
#    shutil.rmtree(homepath)
#    os.makedirs(homepath[:-1])

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

# genData(15)

def init_pool_process():
    np.random.seed()

if __name__ == '__main__':
    # multiprocessing case
    print('Process Starting')
    startTime = time.time()
    amount = Config.amountOfData
    pool_obj = multiprocessing.Pool(initializer=init_pool_process)
    pool_obj.map(genData, range(0,amount))
    pool_obj.close()
    endTime = time.time()

    print('Finish Running')
    print('Average Time: ' + str((endTime - startTime)/amount))




