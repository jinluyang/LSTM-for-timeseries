# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 05:29:34 2017

@author: root
"""
#import tensorflow as tf
import pandas as pd

import numpy as np

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, model_from_json


matrix=pd.read_csv('matrix.csv',index_col=0)
a = matrix


def loss(df1,df2):
    T=len(df1)
    n=len(df1.columns)
    L=0
    for t in range(T):
        for i in range(n):
            L=L+abs(df1.iloc[t][i]-df2.iloc[t][i])/(df1.iloc[t][i]+df2.iloc[t][i])
    return L/n/T
    
def create_interval_dataset(dataset, look_back=14,limitsize=30000):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    dataX, dataY = [], []
    j=limitsize
    for col in dataset.columns:
        for i in range(len(dataset) - look_back):
            dataX.append(dataset.iloc[i:i+look_back][col])
            dataY.append(dataset.iloc[i+look_back][col])
            j-=1
        if j<0:
            break
    return np.asarray(dataX), np.asarray(dataY)
    


        
lookback = 14
x ,y = create_interval_dataset(a ,lookback)
print x.shape
X=np.reshape(x,(x.shape[0],1,x.shape[1]))
#nn = NeuralNetwork()
#nn.NN_model(X,y)
model = Sequential()
model.add(LSTM(200, input_dim=lookback))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X,y,nb_epoch=80,batch_size=256)
#predict
x2=a[-14:]
x2=x2.values
for i in range(14):
    print i
    x2=x2.T
    x3=np.reshape(x2,(x2.shape[0],1,x2.shape[1]))
    te=model.predict(x3)
    te=np.reshape(te,te.shape[0])
    #print te,te.shape
    result = np.vstack((x2.T,te))
    x2=result[1:]

print x2,x2.shape
result = pd.DataFrame(x2.T.astype(int))
#result.columns=['day_%s'%i for i in range(1,15)]
result.index=matrix.columns
result.sort_index(inplace=True)
result=result.applymap(lambda x:x if x>0 else 0)
result.to_csv('result.csv',header=None)
model.save('model2.h5')
   # xx=np.reshape(xx,(xx.shape))
