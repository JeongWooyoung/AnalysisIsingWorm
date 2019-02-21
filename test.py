# coding: utf-8
import numpy as np

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    file_cnt=2
    # temperature = ['1.000', '1.500', '2.000', '2.500', '3.000', '3.500', '4.000', '4.500', '5.000']
    temperature = ['2.500']
    s2_train_input, s4_train_target, s2_test_input, s4_test_target = fh.getWormData(file_cnt, temperature)

    predicts = np.array(fh.loadTxT(fh.getStoragePath()+'results/evaluations_result.txt'))
    predicts = predicts.reshape(predicts.shape[0], 1, 1)
    fh.displayData(predicts, 'Predicts')
    eh.showScatter([s2_train_input, s2_test_input], [s4_train_target, predicts])
