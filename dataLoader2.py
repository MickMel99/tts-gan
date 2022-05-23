# -*- coding: utf-8 -*-
""" 
Example usage:  
x_train, y_train, x_test, y_test = spx_load_dataset()

data range: 1st Jan 2015 -> 1st Jan 2022

what i want= x_train, y_train, x_test, y_test

"""


from re import X
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

# batch size of 10
# patch size of 51
# 1057/50 = 21.14
# 706/50 = 14.12
patch_size = 52
batch_size = 10
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
def transform_data():
    data_df = stock_dl(start,end,stock)
    data_df.loc[:,'label'] = 1
    data_df.drop(labels='Date', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data_df.iloc[:,:-1],data_df.iloc[:,-1],
    test_size=.4, train_size=.6, random_state=12, shuffle=False, stratify=None)
    print(f'shape of X_train / test: {X_train.shape}{X_test.shape}')
    print(f'shape of y_train / test: {y_train.shape} {y_test.shape}')
    print('Reshaping')   

    X_train = X_train.to_numpy()
    X_train = X_train[:, :, np.newaxis]
    X_train = np.transpose(X_train, (1, 2, 0))
    print(f'shape of X_train / test: {X_train.shape}{X_test.shape}')
    print(f'shape of y_train / test: {y_train.shape} {y_test.shape}')

    return X_train, X_test, y_train, y_test


x_train, x_test, y_train, y_test = transform_data()

x_train=x_train.to_numpy()
x_train = x_train[:, :, np.newaxis]
x_train = np.transpose(x_train, (1, 2, 0))

x_train = x_train[:,:,:,:-1]
z = np.array_split(x_train, 21)
z[1].shape
z

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, x_train.shape[2])
x_train = x_train[:,:,:,:-1]

print(z)
# (10, 6, 1, 1763)
def patches_batches(x):
    x = np.array(x)
    x = x[:, :, np.newaxis]
    x = x.reshape(batch_size, x.shape[1], 1, 1763)

    return x

print(patches_batches(x_test).shape)


