# coding: utf-8
import numpy as np
import tensorflow as tf

import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from datetime import datetime

import evaluation_handler as eh

#########################################################################################################
################################### Retain RNN ##########################################################
class LSTM(object):
    def __init__(self, args):
        self.args = args
    def train(self, data, target):
        data_size = data.shape[0]
        train_count = int(np.ceil(data_size / self.args.batch_size))

        sh = np.arange(data.shape[0])
        np.random.shuffle(sh)

        train_data = data[sh]
        target = target[sh]

        test_data = train_data[:100]
        test_target = target[:100]

        start = datetime.now()
        for i in range(self.args.num_epochs):
            losses = []
            for t in range(train_count):
                s = self.args.batch_size * t
                e = s + self.args.batch_size

                train_data_ = train_data[s:e]
                train_target_ = target[s:e]

                _, loss = self.sess.run((self.train_step, self.loss), feed_dict={self.input: train_data_
                                                                    , self.target: train_target_
                                                                    , self.KeepProbCell: self.keep_prob_cell
                                                                    , self.KeepProbLayer: self.keep_prob_layer})
                losses.append(np.mean(np.nan_to_num(loss)))
            if i%10 == 9:
                predicts, rmse = self.sess.run(self.prediction, feed_dict={self.input: test_data, self.KeepProbCell: 1, self.KeepProbLayer: 1})
                accuracy, precision, recall, f1 = eh.evaluatePredictions(test_target, predicts)
                print('=====================================================================================================================================================')
                print('epoch %d: loss %03.5f rmse: %03.5f accuracy : %.4f, precision : %.4f, recall : %.4f, f1-measure : %.4f' % (i+1, np.mean(losses), accuracy, precision, recall, f1))
                print(datetime.now()-start)
                print('=====================================================================================================================================================')
                start = datetime.now()

    def predict(self, data):
        predicts, rmse = self.sess.run((self.prediction, self.rmse), feed_dict={self.input: data, self.KeepProbCell: 1, self.KeepProbLayer: 1})
        return predicts, rmse

    def init_session(self):
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)

    def generateModel(self, name, input, output_size, activation, n_layers, layer_size, reuse):
        def rnn_cell(): return tf.contrib.rnn.BasicRNNCell(layer_size, reuse=reuse)
        with tf.variable_scope('LSTM_'+name, reuse=reuse):
            Cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(rnn_cell(), input_keep_prob=self.KeepProbCell) for _ in range(n_layers)]
                , state_is_tuple=True)
            Cell = tf.contrib.rnn.DropoutWrapper(Cell, input_keep_prob=self.KeepProbLayer)

            # Create RNN
            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of RNNCells containing a
            reverse_input = tf.reverse(input, axis=[1], name='reverse_input')
            Output, State = tf.nn.dynamic_rnn(Cell, reverse_input, dtype=tf.float32)
            h = tf.reverse(Output, axis=[1])

            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.)
            w = tf.get_variable("w", [h.get_shape()[2], output_size], initializer=w_init)
            b = tf.get_variable("b", [output_size], initializer=b_init)

            h = tf.transpose(h, (1, 0, 2))

            e = []
            for i in range(h.get_shape()[0]) :
                e.append(activation(tf.matmul(h[i], w) + b)*0.5)
        return tf.transpose(e, (1, 0, 2))

    def generateModels(self, input_size, output_size, step_size):
        tf.reset_default_graph()

        self.input = tf.placeholder(tf.float32, [None, step_size, input_size])
        self.target = tf.placeholder(tf.float32, [None, output_size])

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = tf.placeholder(tf.float32, [])
        self.KeepProbLayer = tf.placeholder(tf.float32, [])

        # Embedding
        w_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("wv", [step_size, input_size], initializer=w_init)
        self.v = tf.multiply(w, self.input)

        # generating alpha values
        # rnn for α
        self.alpha = self.generateModel(name='layers', input=self.v, output_size=1
                                                , activation=tf.nn.softmax, n_layers=self.args.n_layers_a
                                                , layer_size=self.args.n_hidden_a, reuse=False)

        # generating c
        t = self.alpha
        self.c = tf.reduce_sum(t, 2)

        # generating y
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)
        w = tf.get_variable("wy", [self.c.get_shape()[1], output_size], initializer=w_init)
        b = tf.get_variable("by", [output_size], initializer=b_init)

        self.prediction = tf.nn.softmax(tf.matmul(self.c, w)+b)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.y)
        # self.loss = -tf.reduce_mean(tf.reduce_mean(self.target*tf.log(self.y)+(1-self.target)*tf.log(1-self.y), 1))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.target - self.prediction)))
        self.train_step = self.optimizer.minimize(self.loss)

        self.init_session()
