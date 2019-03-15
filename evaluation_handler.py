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
import matplotlib.pyplot as plt
import arguments

storage_path = fh.getStoragePath()
def evaluations(args, data, targets):
    if not type(data).__module__==np.__name__: data = np.array(data)
    if not type(targets).__module__==np.__name__: targets = np.array(targets)

    kf = KFold(n_splits=10, shuffle=False, random_state=0)
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
        fh.saveTxT(predicts.reshape(predicts.shape[0], 1), 'results/fold/%d'%(i+1))

    return results

def evaluations2(args, train_input, train_targets, test_input, test_targets, eval_file_name='evaluations_result'):
    if not type(train_input).__module__==np.__name__: train_input = np.array(train_input)
    if not type(train_targets).__module__==np.__name__: train_targets = np.array(train_targets)
    if not type(test_input).__module__==np.__name__: test_input = np.array(test_input)
    if not type(test_targets).__module__==np.__name__: test_targets = np.array(test_targets)

    results = []
    lstm = lh.LSTM(args)
    lstm.generateModels(train_input.shape[1], train_targets.shape[1], train_input.shape[2])
    loss = lstm.train(data=train_input, target=train_targets)

    rmse = lstm.evaluation(test_input, test_targets)
    results.append([loss, rmse])
    train_predicts = np.array(lstm.predict(train_input))
    fh.saveTxT(train_predicts.reshape(train_predicts.shape[0], 1), 'results/train_predicts_%s_%d_%d_%d'%(eval_file_name, args.n_layers, args.n_hidden, args.file_cnt))
    fh.displayData(train_predicts, 'Train Predicts')
    test_predicts = np.array(lstm.predict(test_input))
    fh.saveTxT(test_predicts.reshape(test_predicts.shape[0], 1), 'results/test_predicts_%s_%d_%d_%d'%(eval_file_name, args.n_layers, args.n_hidden, args.file_cnt))
    fh.displayData(test_predicts, 'Test Predicts')

    # showScatter([train_input, test_input, test_input], [train_targets, test_targets, predicts])
    saveScatter([train_input, test_input, test_input, train_input], [train_targets, test_targets, test_predicts, train_predicts], args=args)
    # showScatter(test_input, test_targets, predicts)

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


def showScatter(Xs, Ys):
    cnt = len(Xs)
    # fig, axs = plt.subplots(1, cnt, figsize=(10, 10))
    #
    # if cnt == 1:
    #     f1_data = np.concatenate((Xs[0], Ys[0]), axis=2)
    #     f1_data = f1_data.reshape(f1_data.shape[0], f1_data.shape[1]*f1_data.shape[2])
    #     f1_data = np.transpose(f1_data, (1, 0))
    #     axs.scatter(f1_data[0], f1_data[1])
    # else:
    #     for i, (x, y) in enumerate(zip(Xs, Ys)):
    #         f1_data = np.concatenate((x, y), axis=2)
    #         f1_data = f1_data.reshape(f1_data.shape[0], f1_data.shape[1]*f1_data.shape[2])
    #         f1_data = np.transpose(f1_data, (1, 0))
    #
    #         axs[i].scatter(f1_data[0], f1_data[1])
    colors = ['green', 'blue', 'yellow', 'red', 'black']
    for i, (x, y) in enumerate(zip(Xs, Ys)):
        plt.scatter(x, y, color=colors[i])
    plt.show()
def saveScatter(Xs, Ys, args=None):
    cnt = len(Xs)
    # fig, axs = plt.subplots(1, cnt, figsize=(10, 10))
    #
    # if cnt == 1:
    #     f1_data = np.concatenate((Xs[0], Ys[0]), axis=2)
    #     f1_data = f1_data.reshape(f1_data.shape[0], f1_data.shape[1]*f1_data.shape[2])
    #     f1_data = np.transpose(f1_data, (1, 0))
    #     axs.scatter(f1_data[0], f1_data[1])
    # else:
    #     for i, (x, y) in enumerate(zip(Xs, Ys)):
    #         f1_data = np.concatenate((x, y), axis=2)
    #         f1_data = f1_data.reshape(f1_data.shape[0], f1_data.shape[1]*f1_data.shape[2])
    #         f1_data = np.transpose(f1_data, (1, 0))
    #
    #         axs[i].scatter(f1_data[0], f1_data[1])
    colors = ['green', 'blue', 'yellow', 'red', 'black']
    for i, (x, y) in enumerate(zip(Xs, Ys)):
        plt.scatter(x, y, color=colors[i])
    if args is None:
        args = arguments.parse_args()
    fig_path = fh.getStoragePath()+'Figures/'
    fh.makeDirectories(fig_path)
    plt.savefig(fig_path+'Layer_%d_Hidden_%d_Epoch_%d.png'%(args.n_layers, args.n_hidden, args.num_epochs))
    plt.clf()