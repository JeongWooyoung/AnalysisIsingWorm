# coding: utf-8
import csv, os, shutil
import numpy as np
from PIL import Image

def loadData():
    lines = loadCSV(getStoragePath()+'/resources/IAPS.csv')
    lines = np.array(lines)
    # images, images_data = loadImages(lines[:,0])
    labels = np.argmax(lines[:,2:], axis=1)
    # return images, images_data, labels
    return lines[:,0], labels

#########################################################################################################
######################################### CSV ###########################################################

def saveCSV(data, file_name, column_sec=[]):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+"/resources/"+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        os.makedirs(directory)

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
    else : path = getStoragePath()+"/resources/"+file_name.replace('\\','/').replace(":", "")
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
######################################### IMAGE #########################################################
def loadImages(file_list, shape=(28, 28, 1)):
    images_data = np.array([])
    width, height, kernel = shape
    if len(file_list)>0:
        for file_name in file_list:
            img = loadJPG(file_name, height=height, width=width)
            if shape[2] == 1:
                img = img.convert('1')
            img_w, img_h = img.size
            image_data = np.array(img.getdata(), dtype=np.int_)
            if img_w == width and img_h == height: image_data = image_data
            else:
                if img_w > width:
                    image_data = image_data.reshape(img_h, img_w, kernel)[:,:width].reshape(img_h*width*kernel)
                    img_w = width
                elif img_w < width:
                    image_data = np.concatenate((image_data.reshape(img_h, img_w, kernel), np.ones((img_h, width-img_w, kernel))*255), axis=1).reshape(img_h*width*kernel)
                    img_w = width
                if img_h > height:
                    image_data = image_data.reshape(img_h, img_w, kernel)[:height].reshape(height * img_w * kernel)
                elif img_h < height:
                    image_data = np.concatenate((image_data.reshape(img_h, img_w, kernel), np.ones((height-img_h, img_w, kernel))*255), axis=0).reshape(height*width*kernel)

            images_data = np.concatenate((images_data, image_data), axis=0)
        return images_data.reshape(len(file_list), height, width, kernel), images_data.reshape(len(file_list), height*width*kernel)
    else: None
def loadJPG(file_name, height=28, width=28):
    if ".jpg" not in file_name.lower(): file_name = file_name+".jpg"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+"/resources/resized_images/%dx%d/"%(height, width)+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None

    return Image.open(path)

#########################################################################################################
def getStoragePath():
    StoragePath = os.getcwd().replace('\\', '/')
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

