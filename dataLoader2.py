# -*- coding: utf-8 -*-
""" 
Example usage:  
x_train, y_train, x_test, y_test = spx_load_dataset()

data range: 1st Jan 2015 -> 1st Jan 2022


"""


#from re import X
from math import floor
from re import X
from tkinter import N
import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import datetime
from tabulate import tabulate # for verbose tables
import os.path
from os import path
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)


start = datetime.datetime(2015,1,1)
end = datetime.datetime(2022,1,1)
stock = "^GSPC"

def stock_dl(start,end,stock):
    if path.exists('./SP500_Daily.csv'):
        df_daily = pd.read_csv('./SP500_Daily.csv')
        print('reading from file path')
    else:        
        df_daily = yf.download(stock, start, end, interval='1d')
        df_daily.to_csv("./SP500_Daily.csv")#
        print('downloading file')
        return df_daily
    return df_daily

df = stock_dl(start,end,stock)
# len of df = 1763  
# Open High Low Close Adj Close Volume Label

def sho_table(x):
    headers = ("Raw data","shape", "object type", "data type")
    mydata = [(f"{x}", x.shape, type(x), x.dtypes)]
    print(tabulate(mydata, headers=headers))

# labels for real data = 1 one hot encoded
def transform_data(normal = False):
    data_df = stock_dl(start,end,stock)
    data_df.loc[:,'label'] = 1
    data_df.drop(labels='Date', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data_df.iloc[:,:-1],data_df.iloc[:,-1],
    train_size=.6, random_state=12, shuffle=False, stratify=None)
    print(f'shape of X_train / test: {X_train.shape}{X_test.shape}')
    print(f'shape of y_train / test: {y_train.shape} {y_test.shape}')
#    print('Reshaping')   

#    X_train = X_train.to_numpy()
#    X_train = X_train[:, :, np.newaxis]
#    X_train = np.transpose(X_train, (1, 2, 0))
#    print(f'shape of X_train / test: {X_train.shape}{X_test.shape}')
#    print(f'shape of y_train / test: {y_train.shape} {y_test.shape}')
    X_train=X_train.to_numpy()
    X_test=X_test.to_numpy() 
    y_train=y_train.to_numpy() 
    y_test=y_test.to_numpy()

    X_train = X_train.reshape(7, X_train.shape[1], -1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])
    X_train = X_train[:,:,:,:-1]

    X_test = X_test.reshape(2, X_test.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
    X_test = X_test[:,:,:,:-1]
    print(f'shape of X_train / test: {X_train.shape}{X_test.shape}')
    print(f'shape of y_train / test: {y_train.shape} {y_test.shape}')

    def _normalize(epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,0,:] = _normalize(epochs[i,j,0,:])
#                 epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])

        return epochs

    if normal :
        print('Normalising X test and train')
        X_train = normalization(X_train)
        X_test = normalization(X_test)
    return X_train, X_test, y_train, y_test


x_train, x_test, y_train, y_test = transform_data(normal=True)





x_train.shape
(np.transpose(x_train , (0, 2, 1,3))).shape
