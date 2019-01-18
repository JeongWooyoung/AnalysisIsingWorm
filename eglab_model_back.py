import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def MinMaxScaler(data):
        ''' Min Max Normalization
        Parameters
        ----------
        data : numpy.ndarray
            input data to be normalized
            shape: [Batch size, dimension]
        Returns
        ----------
        data : numpy.ndarry
            normalized data
            shape: [Batch size, dimension]
        References
        ----------
        .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        '''
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

class LSTM : 
    # train Parameters
    timesteps = seq_length = 100
    input_dim = 2
    #timesteps = seq_length = 7
    #input_dim = 5
    
    hidden_dim = 10
    output_dim = 1
    learing_rate = 0.01
    iterations = 1000

    # Open, High, Low, Volume, Close
    #xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
    xy = np.loadtxt('./data/data/history_N256_T4.900.dat',delimiter=' ')
    #xy = xy[::-1]  # reverse order (chronically ordered)
    xy = xy[::-1,1:3]
    
    xy = MinMaxScaler(xy)
    x = xy
    y = xy[:,[-1]]  # Close as label
    
    # build a dataset
    dataX = []
    dataY = []
    
    for i in range(0, int(len(y)/seq_length) - seq_length):
    #for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    # train/test split
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

    def __init__(self):

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # GPU 사용 파라미터
        # Initialize the variables (i.e. assign their default value)
        # Run the initializer
        self.rmse, self.loss, self.predict, self.accuracy = self.build_model()

        self.saver, self.session = self.init_session(True)

        self.writer = tf.summary.FileWriter('logs', self.session.graph)

        tf.summary.scalar("Loss/Cost", self.loss)
        self.summary = tf.summary.merge_all()

        print(self.summary)

    def init_session(self, gpu = False):
        # with tf.device("/cpu:0"):
        saver = tf.train.Saver()

        if gpu :
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.8
            session = tf.InteractiveSession(config=config)
        else :
            session = tf.InteractiveSession()

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print ("새로운 모델을 생성하는 중 입니다.")
            session.run(tf.global_variables_initializer())

        return saver, session

    '''
    To classify images using a recurrent neural network, we consider every image
    row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
    handle 28 sequences of 28 steps for every sample.
    '''

    def build_model(self):

        outputs, _states = self.RNN()
        with tf.device("/gpu:0"):
            self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output
            # cost/loss
            loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
            # optimizer
            optimizer = tf.train.AdamOptimizer(self.learing_rate)
            self.train = optimizer.minimize(loss, global_step= self.global_step)
        
            # RMSE
            self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
            self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
        
            rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(self.predictions, self.output_dim), tf.argmax(self.Y, self.output_dim))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return rmse, loss, correct_pred, accuracy

    def data_train(self):
        for i in range(self.iterations):
            # Run optimization op (backprop)
            with tf.device("/gpu:0"):
                _, step_loss = self.session.run([self.train, self.loss], feed_dict={
                                    self.X: self.trainX, self.Y: self.trainY})
                print("[step: {}] loss: {}".format(i, step_loss))                
                self.write_logs( step_loss, i)

        print("Optimization Finished!")
    
    def data_test(self):
        with tf.device("/gpu:0"):
            test_predict = self.session.run( self.Y_pred, feed_dict={ self.X:  self.testX})
            self.rmse = self.session.run( self.rmse, feed_dict={
                        self.targets:  self.testY,  self.predictions:  test_predict})
            print("RMSE: {}".format(self.rmse))
        
        # Plot predictions
        plt.plot(self.testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Predictive Value (Y2)")
        plt.show()

    def write_logs(self, loss, time_step):
        #with tf.device("/gpu:0"):
        #print(self.summary)
        if time_step % 100 == 0:
            summary = self.summary.eval(feed_dict={self.loss:loss})
            self.writer.add_summary(summary, self.global_step.eval())

        if time_step % 10000 == 0:
            self.saver.save(self.session, './model/lstm.ckpt', global_step=time_step)


    def RNN(self):
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell( num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.X, dtype=tf.float32)

        return outputs, states

