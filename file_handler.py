# coding: utf-8
import csv, os, shutil
import numpy as np

#########################################################################################################
######################################### TXT ###########################################################

def saveTxT(data, file_name):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+"/"+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        makeDirectories(directory)

    np.savetxt(fname=path, X=data, delimiter=' ')
    return True

def loadTxT(file_name, column_rows=0):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+"/"+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None

    lines = np.loadtxt(fname=path, delimiter=' ')
    return lines[column_rows:]

#########################################################################################################
######################################### CSV ###########################################################

def saveCSV(data, file_name, column_sec=[]):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+"/"+file_name.replace('\\','/').replace(":", "")
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
    else : path = getStoragePath()+"/"+file_name.replace('\\','/').replace(":", "")
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

