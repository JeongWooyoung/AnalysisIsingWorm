# coding: utf-8
import csv, os, shutil
import numpy as np

headFile = 'head'
tailFile = 'tail'
def getWormData(file_cnt=1, temperature=['1.000']):
    train_path = getStoragePath()+'data/IsingWorm/training_set/'
    test_input_path = getStoragePath()+'data/IsingWorm/evaluation_input/'
    test_target_path = getStoragePath()+'data/IsingWorm/compare_with_prediction/'

    train_input = getPathData(train_path, headFile, file_cnt, temperature)
    train_target = getPathData(train_path, tailFile, file_cnt, temperature)

    test_input = getPathData(test_input_path, headFile, file_cnt, temperature)
    test_target = getPathData(test_target_path, tailFile, file_cnt, temperature)

    displayData(train_input, 'Train Input')
    displayData(train_target, 'Train Target')
    displayData(test_input, 'Test Input')
    displayData(test_target, 'Test Target')

    return train_input, train_target, test_input, test_target
def getPathData(path, file_name, file_cnt=1, temperature=['1.000']):
    data = []
    for t in temperature:
        for i in range(file_cnt):
            tmp = loadTxT(path+'N8_T%s/%s%d'%(t, file_name, i))
            if not tmp is None:
                data += tmp.tolist()
    if len(data) < 1: return None
    data = np.array(data, dtype=np.float_)
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return data[:,:,1:2]

def displayData(data, name='Data'):
    mean = np.mean(data)
    std = np.std(data)
    max = np.max(data)
    min = np.min(data)
    percentile = np.percentile(data, [25, 50, 75])
    print('%s. Avg: %3.5f Std: %3.5f Max: %3.5f Min: %3.5f Percentile(25, 50, 75): %3.5f %3.5f %3.5f Count: %6d'%
          (name, mean, std, max, min, percentile[0], percentile[1], percentile[2], data.shape[0]))

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
    StoragePath = os.getcwd().replace('\\', '/')+'/'
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

