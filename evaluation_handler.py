# coding: utf-8

#########################################################################################################
## Evaluation ###########################################################################################
import numpy as np
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import file_handler as fh
import learn_handler as lh
from datetime import datetime

storage_path = fh.getStoragePath()
def evaluations(args, images_file, target):
    if not type(target).__module__==np.__name__: target = np.array(target)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    shape = [360, 512, 1]
    results = []
    for i, (train_index, test_index) in enumerate(kf.split(images_file)):
        train_images, train_images_data =  fh.loadImages(images_file[train_index], shape=shape)
        train_target = np.argmax(target[train_index], axis=1)

        # cnn.generateModels(shape[0], shape[1], shape[2], target.shape[1])
        cnn = lh.CNN(args)
        cnn.generateModels(shape[0], shape[1], shape[2], target.shape[1])
        timer = datetime.now()
        cnn.train(data=train_images_data, target=train_target)
        train_time = datetime.now()-timer

        test_images, test_images_data = fh.loadImages(images_file[test_index], shape=shape)
        test_target = np.argmax(target[test_index], axis=1)

        e = cnn.evaluation(test_images_data, test_target)
        print('Eval step {0}. accuracy : {1} loss : {2}, training time : {3}'.format(i+1, e['accuracy'], e['loss'], train_time))
        results.append([e['accuracy'], e['loss']])
        fh.clearCaches()

        # predicts = cnn.predict(test_images_data)
        # accuracy, precision, recall, f1 = evaluatePredictions(test_target, predicts)
        #
        # results.append([accuracy, precision, recall, f1])
    return results

def evaluatePredictions(test_labels, predicts):
    # average = 'binary'
    average = 'weighted'
    pos_label = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        accuracy = accuracy_score(y_pred=predicts, y_true=test_labels)
        precision = precision_score(y_pred=predicts, y_true=test_labels, average=average, pos_label=pos_label)
        recall = recall_score(y_pred=predicts, y_true=test_labels, average=average, pos_label=pos_label)
        f1 = f1_score(y_pred=predicts, y_true=test_labels, average=average, pos_label=pos_label)

    return accuracy,precision, recall, f1
