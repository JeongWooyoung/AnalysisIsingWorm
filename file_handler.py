# coding: utf-8
import csv, os, shutil
import numpy as np

headFile = 'head'
tailFile = 'tail'
def getData(file_cnt):
    head_data = []
    tail_data = []
    for i in range(file_cnt):
        head_data += loadTxT(headFile+'%d'%(i+1)).tolist()
        tail_data += loadTxT(tailFile+'%d'%(i+1)).tolist()
    head_data = np.array(head_data, dtype=np.float_)
    tail_data = np.array(tail_data, dtype=np.float_)
    head_data = head_data.reshape(head_data.shape[0], 1, head_data.shape[1])
    tail_data = tail_data.reshape(tail_data.shape[0], 1, tail_data.shape[1])
    s2_train_data, s2_target_data, s4_train_data, s4_target_data = head_data[:,:,1:2], tail_data[:,:,1:2], head_data[:,:,2:3], tail_data[:,:,2:3]

    displayData(s2_train_data, 'S2 Train Data')
    displayData(s2_target_data, 'S2 Target Data')
    displayData(s4_train_data, 'S4 Train Data')
    displayData(s4_target_data, 'S4 Target Data')

    return s2_train_data, s2_target_data, s4_train_data, s4_target_data
def displayData(data, name='Data'):
    mean = np.mean(data)
    std = np.std(data)
    max = np.max(data)
    min = np.min(data)
    percentile = np.percentile(data, [25, 50, 75])
    print('%s. Avg: %3.5f Std: %3.5f Max: %3.5f Min: %3.5f Percentile(25, 50, 75): %3.5f %3.5f %3.5f'%
          (name, mean, std, max, min, percentile[0], percentile[1], percentile[2]))

#########################################################################################################
######################################### TXT ###########################################################

def saveTxT(data, file_name):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        makeDirectories(directory)

    np.savetxt(fname=path, X=data, delimiter=' ')
    return True

def loadTxT(file_name, column_rows=0):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None

    lines = np.loadtxt(fname=path, delimiter=' ')
    return lines[column_rows:]

#########################################################################################################
######################################### CSV ###########################################################

def saveCSV(data, file_name, column_sec=[]):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        makeDirectories(directory)

    csv_file = open(path, "w", newline='\n')
    cw = csv.writer(csv_file, delimiter=',', quotechar='|')

    if len(column_sec) > 0:
        for columns in column_sec:
            cw.writerow(columns)
    for row in data:
        cw.writerow(row)
    csv_file.close()
    return True

def loadCSV(file_name, column_rows=0):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None
    csv_file = open(path, "r")
    cr = csv.reader(csv_file, delimiter=',', quotechar='|')

    lines = []
    for line in cr:
        lines.append(line)

    csv_file.close()

    return lines[column_rows:]

#########################################################################################################
#########################################################################################################
def makeDirectories(directory):
    if '\\' in directory: directory = directory.replace('\\', '/')
    if '/' in directory:
        u_dir = directory[:directory.rfind('/')]
        if not os.path.isdir(u_dir):
            makeDirectories(u_dir)
    if not os.path.isdir(directory):
        os.makedirs(directory)
def getStoragePath():
    StoragePath = os.getcwd().replace('\\', '/')+'/training_set/N64_T2.300_H0.000/'
    return StoragePath

def clearCaches():
    cache_dir = 'C:/Users/Wooyoung/AppData/Local/Temp/'
    if not os.path.isdir(cache_dir) : return
    files = os.listdir(cache_dir)
    for file in files:
        if 'tmp' in file:
            if not os.path.isfile(cache_dir+file) :
                try: shutil.rmtree(cache_dir+file)
                except OSError as e: pass

