# coding: utf-8
import numpy as np
import tensorflow as tf

from datetime import datetime

import file_handler as fh

#########################################################################################################
################################### Retain RNN ##########################################################
class LSTM(object):
    def __init__(self, args):
        self.args = args
        # self.saver = tf.train.Saver()
    def train(self, data, target):
        data_size = data.shape[0]
        train_count = int(np.ceil(data_size / self.args.batch_size))

        sh = np.arange(data.shape[0])
        np.random.shuffle(sh)

        train_data = data[sh]
        target = target[sh]

        test_data = train_data[:1000]
        test_target = target[:1000]

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
            if i%100 == 99:
                predicts, rmse = self.sess.run((self.prediction, self.rmse), feed_dict={self.input: test_data, self.target: test_target, self.KeepProbCell: 1, self.KeepProbLayer: 1})
                # accuracy, precision, recall, f1 = eh.evaluatePredictions(test_target, predicts)
                print('=====================================================================================================================================================')
                # print('epoch %d: loss %03.5f rmse: %03.5f accuracy : %.4f, precision : %.4f, recall : %.4f, f1-measure : %.4f' % (i+1, np.mean(losses), rmse, accuracy, precision, recall, f1))
                print('epoch %d: loss %03.9f rmse: %03.5f' % (i + 1, np.mean(losses), rmse))
                print(datetime.now()-start)
                print('=====================================================================================================================================================')
                fh.saveTxT(predicts.reshape(predicts.shape[0], 1), 'predicts/epoch_%d' % (i + 1))
                start = datetime.now()
        return np.mean(losses)

    def predict(self, data):
        predicts = self.sess.run((self.prediction), feed_dict={self.input: data, self.KeepProbCell: self.args.keep_prob_cell, self.KeepProbLayer: self.args.keep_prob_layer})
        return predicts
    def evaluation(self, data, target):
        rmse = self.sess.run((self.rmse), feed_dict={self.input: data, self.target: target, self.KeepProbCell: 1,self.KeepProbLayer: 1})
        return rmse

    def init_session(self):
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        tf.local_variables_initializer().run(session=self.sess)

    def generateModel(self, name, input, output_size, activation, n_layers, layer_size, reuse):
        def rnn_cell(): return tf.contrib.rnn.BasicLSTMCell(layer_size, reuse=reuse)
        # with tf.variable_scope('LSTM_'+name, reuse=reuse):
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
        return tf.transpose(e, (1, 0, 2), name=name)

    def generateModels(self, input_size, output_size, step_size):
        tf.reset_default_graph()

        self.input = tf.placeholder(tf.float32, [None, step_size, input_size], name='input')
        self.target = tf.placeholder(tf.float32, [None, step_size, output_size], name='target')

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = tf.placeholder(tf.float32, [], name='KeepProbCell')
        self.KeepProbLayer = tf.placeholder(tf.float32, [], name='KeepProbLayer')

        # generating alpha values
        self.alpha = self.generateModel(name='layers', input=self.input, output_size=1
                                                , activation=tf.nn.elu, n_layers=self.args.n_layers
                                                , layer_size=self.args.n_hidden, reuse=False)

        self.prediction = self.alpha
        target = self.target[:,:,-1]
        predict = self.prediction[:,:,-1]
        # target_shape = tf.shape(self.target)
        # predict_shape = tf.shape(self.prediction)
        # target = tf.reshape(self.target, [target_shape[0], target_shape[1]*target_shape[2]])
        # predict = tf.reshape(self.prediction, [predict_shape[0], predict_shape[1]*predict_shape[2]])

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate, name='optimizer')
        self.loss = tf.losses.mean_squared_error(target, predict)
        # self.loss = tf.reduce_mean(tf.square(target - predict))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(target - predict)), name='rmse')
        self.train_step = self.optimizer.minimize(self.loss, name='train_step')

        self.init_session()

    def save(self, model_path):
        self.saver.save(self.sess, model_path, global_step=1000)
    def restore(self, model_path):
        self.init_session()
        self.saver = tf.train.import_meta_graph(model_path+'-1000.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path[:model_path.rfind('/')+1]))

        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.target = graph.get_tensor_by_name("target:0")

        self.keep_prob_cell = self.args.keep_prob_cell
        self.keep_prob_layer = self.args.keep_prob_layer
        self.KeepProbCell = graph.get_tensor_by_name('KeepProbCell:0')
        self.KeepProbLayer = graph.get_tensor_by_name('KeepProbLayer:0')

        self.prediction = graph.get_tensor_by_name('predict:0')
        layer_name = 'layers'
        self.alpha = graph.get_tensor_by_name('LSTM_%s:%s:0'%(layer_name, layer_name))
        # self.emb = graph.get_tensor_by_name('wv:0')
        # self.wy = graph.get_tensor_by_name('wy:0')

        self.optimizer = graph.get_tensor_by_name("optimizer:0")
        self.loss = graph.get_tensor_by_name("loss:0")
        self.rmse = graph.get_tensor_by_name("rmse:0")
        self.train_step = graph.get_tensor_by_name("train_step:0")