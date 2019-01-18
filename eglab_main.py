'''
This script shows how to predict stock prices using a basic RNN
'''

from __future__ import print_function

import tensorflow as tf
import sys

from eglab_model import LSTM

def main(_):
    for filenum in range(1,50):
        train_model = LSTM(filenum)
        #train_model.data_train()
        train_model.data_test()
        tf.get_variable_scope().reuse_variables()

    sys.exit(0)

if __name__ == '__main__':
    tf.app.run()
