# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 05:29:34 2017

@author: root
"""

import pandas as pd

import numpy as np

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, model_from_json
#from keras.callbacks import ModelCheckpoint

#matrix=pd.read_csv('matrix.csv',index_col=0)
matrix=pd.read_csv('m2.csv',index_col=0)

a = matrix[:-14]
b = matrix[-14:]
del matrix

def loss(df1,df2):
    T=len(df1)
    n=len(df1.columns)
    L=0
    mse=0  #mean square error
    for t in range(T):
        for i in range(n):
            L=L+abs(df1.iloc[t][i]-df2.iloc[t][i])/(df1.iloc[t][i]+df2.iloc[t][i])
            mse+=pow(df1.iloc[t][i]-df2.iloc[t][i],2)
    mse=mse/n/T
    L=L/n/T
    print 'mean square error:%s'%mse
    print 'Loss:%s'%L
    return L
    


def create_interval_dataset(dataset, look_back=14,limitsize=2000):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    data = []
    j=limitsize
    for col in dataset.columns:
        for i in range(len(dataset) - look_back):
            data.append(dataset.iloc[i:i+look_back+1][col])
            #dataY.append(dataset.iloc[i+look_back][col])
            j-=1
        if j<0:
            break
    data=np.asarray(data)
    print data
    print data.shape
    np.random.shuffle(data)
    dataX=data[:,:-1]
    dataY=data[:,-1]
    return (dataX), (dataY)
    


class NeuralNetwork():
    def __init__(self, **kwargs):
        self.output_dim = kwargs.get('output_dim', 16)
        self.activation_lstm = kwargs.get('activation_lstm', 'tanh')
        self.activation_dense = kwargs.get('activation_dense', 'linear')
        self.activation_last = kwargs.get('activation_last', 'linear')    # softmax for multiple output
        self.dense_layer = kwargs.get('dense_layer', 13)     # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 13)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model(self, trainX, trainY, ):#testX, testY):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: epect value of test data
        :return: model after training
        """
        print "Training model is LSTM network!"
        input_dim = trainX[1].shape[0]
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfitting
        model.add(LSTM(output_dim=self.output_dim,
                       input_dim=input_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       return_sequences=True))
        for i in range(self.lstm_layer-1):
            model.add(LSTM(output_dim=self.output_dim,
                       input_dim=self.output_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       return_sequences=True))
        model.add(Dense(output_dim=1,
                        input_dim=self.output_dim,
                        activation=self.activation_last))
        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size)#, validation_data=(testX, testY))
        # store model to json file
#        model_json = model.to_json()

        return model
        
lookback = 21
x ,y = create_interval_dataset(a ,lookback)
print x.shape
X=np.reshape(x,(x.shape[0],1,x.shape[1]))
#nn = NeuralNetwork()
#nn.NN_model(X,y)
model = Sequential()
model.add(LSTM(250, input_dim=lookback))#, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#checkpointer = ModelCheckpoint(filepath="./tmp/weights.hdf5", verbose=1, save_best_only=True)

model.fit(X,y,nb_epoch=200,batch_size=2048)

#predict
x2=a[-lookback:]
x2=x2.values
for i in range(14):
    #print i
    x2=x2.T
    x3=np.reshape(x2,(x2.shape[0],1,x2.shape[1]))
    te=model.predict(x3)
    te=np.reshape(te,te.shape[0])
    if i==0:
        print model.evaluate(x3,(b.ix[0]).values)
    result = np.vstack((x2.T,te))
    x2=result[1:]

print x2,x2.shape
result = pd.DataFrame(x2[-14:])
#print result.shape,b.shape

print loss(b,result)
#print 'loss of day 1:...'

#model.save('model.h5')
   # xx=np.reshape(xx,(xx.shape))
