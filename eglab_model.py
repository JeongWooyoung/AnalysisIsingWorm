import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

valid_set_size_percentage = 0 
test_set_size_percentage = 0

rootPath = './training_set/N64_T2.300_H0.000/'
trainFile = 'head'
testFile = 'tail'
ext = '.txt'

modelPath = './models/'

def normalize_data(datas):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    #datas['c1'] = min_max_scaler.fit_transform(datas.c1.values.reshape(-1,1))
    datas['c2'] = min_max_scaler.fit_transform(datas.c2.values.reshape(-1,1))
    datas['c3'] = min_max_scaler.fit_transform(datas.c3.values.reshape(-1,1))
    return datas

def load_data(data, seq_len, isTrain):
    
    data_raw = data.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data)

    if True == isTrain : 
        valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0])) 
        test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]))
        train_set_size = data.shape[0] - (valid_set_size + test_set_size)
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
        y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
        
        x_test = data[train_set_size+valid_set_size:,:-1,:]
        y_test = data[train_set_size+valid_set_size:,-1,:]
        
        return [x_train, y_train, x_valid, y_valid, x_test, y_test]
    else :
        train_set_size = data.shape[0] 
        
        x_test = data[:train_set_size,:-1,:]
        y_test = data[:train_set_size,-1,:]
        
        return [x_test, y_test]


def load_data_from_file(filename):
    # Set Columns from data 
    filedata = pd.read_csv(filename, names=['c1','c2','c3'], header=None,  sep='\s+', index_col='c1')
    # normalize data
    filedata_data = filedata.copy()
    cols = list(filedata_data.columns.values)
    print('filedata_data.columns.values = ',cols)
   
    # normalize data
    df_data_norm = filedata_data.copy()
    #df_data_norm = normalize_data(df_data_norm)

    return df_data_norm

class LSTM : 

    # create train, test data
    seq_len = 20 # choose sequence length

    #x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(test_data,seq_len)
    fileIdx = 0

    x_train = []
    x_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    # parameters
    n_steps = seq_len-1 
    n_inputs = 2 
    n_neurons = 200 
    n_outputs = 2
    n_layers = 2
    learning_rate = 0.001
    batch_size = 50
    n_epochs = 100 

    train_set_size = ()
    test_set_size = ()

    clayers = 2
    
    index_in_epoch = 0

    def __init__(self, fileIdx):
        
        self.fileIdx = fileIdx

        #initialize
        self.train_data= load_data_from_file(rootPath+trainFile+str(fileIdx)+ext)
        self.test_data= load_data_from_file(rootPath+testFile+str(fileIdx)+ext)
        
        self.x_train, self.y_train, self.x_valid, self.y_valid, _, _ = load_data(self.train_data,self.seq_len, True)
        self.x_test, self.y_test = load_data(self.test_data,self.seq_len, False)

        self.train_set_size = self.x_train.shape[0]
        self.test_set_size = self.x_test.shape[0]

        self.perm_array  = np.arange(self.x_train.shape[0])
        np.random.shuffle(self.perm_array)

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # GPU 사용 파라미터
        # Initialize the variables (i.e. assign their default value)
        # Run the initializer
        self.loss, self.training_op, self.outputs, self.average = self.build_model()

        self.saver, self.session = self.init_session(True)

        self.writer = tf.summary.FileWriter('logs', self.session.graph)

        tf.summary.scalar("Loss/Cost", self.loss)
        #tf.summary.scalar("Average", self.average)
        self.summary = tf.summary.merge_all()

        print(self.summary)
    
    def viewDatatoTrain(self, col1, col2):
        plt.figure(figsize=(15, 5))
        plt.plot(col1, color='red', label='c2')
        plt.plot(col2, color='blue', label='c3')

        plt.title('Representation of normalized data from data sets')
        plt.xlabel('Time Steps')
        plt.ylabel('C2, C3')

    def viewResultTrainedData(self, pred_TestDatas):
        # coltype, 0 = C2 value, 1 = C3 value
        coltype = 0
        ## show predictions
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)

        plt.plot(np.arange(self.y_test.shape[0]),
                self.y_test[:,coltype], color='red', label='test target')

        plt.plot(np.arange(pred_TestDatas.shape[0]),
                pred_TestDatas[:,coltype], color='blue', label='test prediction')

        plt.title('C2 Values Prediction')
        plt.xlabel('Time Step')
        plt.ylabel(self.colsname(coltype))
        plt.legend(loc='best')

        coltype = 1
        plt.subplot(1,2,2)
        plt.plot(np.arange(self.y_test.shape[0]),
                self.y_test[:,coltype], color='orange', label='test target')                 

        plt.plot(np.arange(pred_TestDatas.shape[0]),
                pred_TestDatas[:,coltype], color='green', label='test prediction')

        plt.title('C3 Values Prediction')
        plt.xlabel('Time Step')
        plt.ylabel(self.colsname(coltype))
            
        plt.legend(loc='best')
        plt.show()

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

        ckpt = tf.train.get_checkpoint_state(modelPath +'model_'+str(self.fileIdx))
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
            stacked_rnn_outputs = tf.reshape(outputs, [-1, self.n_neurons]) 
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs, reuse=tf.AUTO_REUSE)
            outputs = tf.reshape(stacked_outputs, [-1, self.n_steps, self.n_outputs])
            outputs = outputs[:,self.n_steps-1,:] # keep only last output of sequence

            average = tf.reduce_mean(outputs - self.Y)                                                        
            loss = tf.reduce_mean(tf.square(outputs - self.Y)) # loss function = mean squared error 
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) 
            training_op = optimizer.minimize(loss)

        return loss, training_op, outputs, average

    # function to get the next batch
    def get_next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        
        if self.index_in_epoch > self.x_train.shape[0]:
            np.random.shuffle(self.perm_array) # shuffle permutation array
            start = 0 # start next epoch
            self.index_in_epoch = batch_size
            
        end = self.index_in_epoch
        return self.x_train[self.perm_array[start:end]], self.y_train[self.perm_array[start:end]]

    def data_train(self):
        for iteration in range(int(self.n_epochs*self.train_set_size/self.batch_size)):
            x_batch, y_batch = self.get_next_batch(self.batch_size) # fetch the next training batch 
            self.session.run(self.training_op, feed_dict={self.X: x_batch, self.Y: y_batch}) 
            # Run optimization op (backprop)
            with tf.device("/gpu:0"):
                if iteration % int(5*self.train_set_size/self.batch_size) == 0:
                    mse_train = self.loss.eval(feed_dict={self.X: self.x_train, self.Y: self.y_train}) 
                    #m_train = self.average.eval(feed_dict={self.X: self.x_valid, self.Y: self.y_train}) 
                    mse_valid = self.loss.eval(feed_dict={self.X: self.x_valid, self.Y: self.y_valid}) 
                    print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(
                        iteration*self.batch_size/self.train_set_size, mse_train, mse_valid))
                    
                    self.write_logs( mse_train, iteration)

        print("Optimization Finished!")
    
    def colsname(self, idx):
        return {0 : "C2 Values", 
                1 : "C3 Values"}.get(idx,"No Columns")
    
    def data_test(self):
        with tf.device("/gpu:0"):

            y_train_pred = self.session.run(self.outputs, feed_dict={self.X: self.x_train})
            y_valid_pred = self.session.run(self.outputs, feed_dict={self.X: self.x_valid})
            y_test_pred = self.session.run(self.outputs, feed_dict={self.X: self.x_test})
            
            ## show predictions
            self.viewResultTrainedData(y_test_pred)
            
            filename = "./results/predict_tail"+str(self.fileIdx)+".txt"
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            predResult = open(filename, 'w')
            
            for i in range(0, y_test_pred.shape[0]):
                data = "{0} {1} {2}\n".format(i, float(y_test_pred[i:i+1,0]), float(y_test_pred[i:i+1,1]))
                predResult.write(data)

            predResult.close()

            corr_price_development_train = np.sum(np.equal(np.sign(self.y_train[:,1] - self.y_train[:,0]),
                        np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / self.y_train.shape[0]
            corr_price_development_valid = np.sum(np.equal(np.sign(self.y_valid[:,0]-self.y_valid[:,0]),
                        np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / self.y_valid.shape[0]
            corr_price_development_test = np.sum(np.equal(np.sign(self.y_test[:,0]-self.y_test[:,0]),
                        np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / self.y_test.shape[0]
            
            print('correct sign prediction - C2 and C3 for train/valid/test: %.2f/%.2f/%.2f'%(
                corr_price_development_train, corr_price_development_valid, corr_price_development_test))

    def write_logs(self, loss, time_step):
        #with tf.device("/gpu:0"):
        summary = self.summary.eval(feed_dict={self.loss:loss})
        self.writer.add_summary(summary, self.global_step.eval())
        self.saver.save(self.session, modelPath +'model_'+str(self.fileIdx)+'/lstm.ckpt', global_step=time_step)

    def RNN(self):
        # Define a lstm cell with tensorflow
        '''lstm_cell = tf.contrib.rnn.BasicLSTMCell( num_units=self.n_neurons, state_is_tuple=True, activation=tf.tanh)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.X, dtype=tf.float32)'''
        cell_layers = [tf.contrib.rnn.BasicLSTMCell( num_units=self.n_neurons, activation=tf.nn.elu) for layer in range(self.n_layers)]
        # Get lstm cell output
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cell_layers)
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.X, dtype=tf.float32)

        return outputs, states