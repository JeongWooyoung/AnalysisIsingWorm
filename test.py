# coding: utf-8
import numpy as np

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    file_cnt=1
    n_layers=3
    n_hidden=30
    # temperature = ['1.000', '1.500', '2.000', '2.500', '3.000', '3.500', '4.000', '4.500', '5.000']
    temperature = ['2.500']
    train_input, train_target, test_input, test_target = fh.getWormData(file_cnt, temperature)

    train_predicts = np.array(fh.loadTxT(fh.getStoragePath()+'results/train_predicts_evaluations_result_%d_%d_%d.txt')%(n_layers, n_hidden, file_cnt))
    test_predicts = np.array(fh.loadTxT(fh.getStoragePath()+'results/test_predicts_evaluations_result_%d_%d_%d.txt')%(n_layers, n_hidden, file_cnt))
    train_predicts = train_predicts.reshape(train_predicts.shape[0], 1, 1)
    test_predicts = test_predicts.reshape(test_predicts.shape[0], 1, 1)
    fh.displayData(train_predicts, 'Train Predicts')
    fh.displayData(test_predicts, 'Test Predicts')
    eh.showScatter([train_input, test_input, test_input, train_input], [train_target, test_target, test_predicts, train_predicts])
    # colors = ['green', 'blue', 'yellow', 'red', 'black']