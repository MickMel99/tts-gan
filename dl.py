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

class load_dataset(Dataset):
    def __init__(self,
        is_normalize = False,
        one_hot_encode = True, 
        data_mode = 'Train',):
        self.is_normalize = is_normalize
        self.one_hot_encode = one_hot_encode
        self.data_mode - data_mode
        

        if (self.one_hot_encode):
            y_train = self.to_categorical(y_train, num_classes=1)
            y_test = self.to_categorical(y_test, num_classes=1)
            print("After one-hot encoding")
            print("x/y_train shape ",x_train.shape,y_train.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)