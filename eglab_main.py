# coding: utf-8
'''
This script shows how to predict stock prices using a basic RNN
'''

# from __future__ import print_function
#
# import tensorflow as tf
# import sys
#
# from eglab_model import LSTM
#
# def main(_):
#     for filenum in range(1,50):
#         train_model = LSTM(filenum)
#         #train_model.data_train()
#         train_model.data_test()
#         tf.get_variable_scope().reuse_variables()
#
#     sys.exit(0)
#
# if __name__ == '__main__':
#     tf.app.run()

import numpy as np

import arguments
import file_handler as fh
import evaluation_handler as eh

trainFile = 'head'
targetFile = 'tail'
if __name__ == '__main__':
    i=1
    train_data = np.array(fh.loadTxT(trainFile+'%d'%(i)), dtype=np.float_)
    target_data = np.array(fh.loadTxT(targetFile+'%d'%(i)), dtype=np.float_)
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    target_data = target_data.reshape(target_data.shape[0], 1, target_data.shape[1])

    args = arguments.parse_args()
    s2_train_data = train_data[:,:,1:2]
    s2_target_data = target_data[:,:,1:2]
    print("susceptibility")
    s2_result = eh.evaluations(args, s2_train_data, s2_target_data)
    s2_result = np.array(s2_result)
    print('=====================================================================================================================================================')
    print('S2 Average: loss %03.9f rmse: %03.5f' % (i + 1, np.mean(s2_result[:,1]), np.mean(s2_result[:,0])))
    print('=====================================================================================================================================================')

    s4_train_data = train_data[:,:,2:3]
    s4_target_data = target_data[:,:,2:3]
    print("susceptibility4")
    eh.evaluations(args, s4_train_data, s4_target_data)
    print('=====================================================================================================================================================')
    print('S4 Average: loss %03.9f rmse: %03.5f' % (i + 1, np.mean(s2_result[:,1]), np.mean(s2_result[:,0])))
    print('=====================================================================================================================================================')
