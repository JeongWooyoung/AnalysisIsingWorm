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

storage_path = fh.getStoragePath()
def evaluations(args, data, targets):
    if not type(data).__module__==np.__name__: data = np.array(data)
    if not type(targets).__module__==np.__name__: targets = np.array(targets)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    shape = data.shape
    results = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_data, train_target = data[train_index], targets[train_index]

        lstm = lh.LSTM(args)
        lstm.generateModels(shape[1], targets.shape[1], shape[2])
        loss = lstm.train(data=train_data, target=train_target)

        test_data, test_target = data[test_index], targets[test_index]

        rmse = lstm.evaluation(test_data, test_target)
        results.append([loss, rmse])
        predicts = np.array(lstm.predict(test_data))
        fh.saveTxT(predicts.reshape(predicts.shape[0], 1), 'fold/%d'%(i+1))

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
