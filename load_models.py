#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:23:50 2023

@author: tristan
"""

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import finance_functions as ff
# import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
import keras

# import xgboost_model as xgbm
import stationary_model as sm
import LSTM_model as lstmm


"""
Implementation of three algorithms to predict next day's Adjusted closed price
of a stock, restricting the model to the last "lookBack" days.

1: Simplest assumption, price of tomorrow is equal to the price of today
2: XGB model, using information about the date, the price and exchange volume of the last "lookBack" days
3: LSTM model, using information about the price and exchange volume of the last "lookBack" days

(TEMPORARY)
Compute the predicted price of tomorrow by taking the ensemble average of the different models
"""



stock="GOOG"  #From those available in dataset/
lookBack=1 #Number of days to lookback into

#Load data 
df=pd.read_csv("dataset/GOOG.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close','Volume'],na_values=['nan'])
df=df.dropna()

#Compute the daily returns of both the Adjusted close price and the exchange volume
df=ff.data_frame_daily_returns(df)
# df=df[:]

# Split of the data into the training and testing sets.
# (TEMPORARY) Needs to be consitent with the _model files
NDays=len(df)
NTrain=int(0.8*NDays)
NTest=NDays-NTrain-lookBack

y_test=df['Adj Close'][-NTest:]


print(lookBack)

"""
Stationary model
"""
print("Stationary model")

y_sm=sm.create_feature(df, label='DR',lookBack=lookBack)
y_sm_predict=y_sm[NTrain:].copy().to_numpy()
print("MSE: ",mean_squared_error(y_sm_predict, y_test.to_numpy()))
print("MAE: ",mean_absolute_error(y_sm_predict, y_test.to_numpy()))



"""
XGBoost
"""
# print("\nXGBoost model")

# model_xgb = xgb.XGBRegressor()
# model_xgb_file=f"dataset/model_{stock}_xgboost_lookback_{lookBack}.json"

# if not os.path.isfile(model_xgb_file) or True:
#     print("Train")
#     xgbm.train(lookBack, stock)
# model_xgb.load_model(model_xgb_file)



# X_xgb,y_xgb=xgbm.create_features(df, label='Adj Close',lookBack=lookBack)

# X_xgb_train=X_xgb[:NTrain].copy()
# y_xgb_train=y_xgb[:NTrain].copy()
# X_xgb_test=X_xgb[NTrain:].copy()
# y_xgb_test=y_xgb[NTrain:].copy()

# y_xgb_predict=model_xgb.predict(X_xgb_test)

# print("MSE: ",mean_squared_error(y_test.to_numpy(), y_xgb_predict))
# print("MAE: ",mean_absolute_error(y_test.to_numpy(), y_xgb_predict))

# print(min(y_test),y_test.mean(),max(y_test))
# print(min(y_xgb_predict),y_xgb_predict.mean(),max(y_xgb_predict))

"""
LSTM model
"""


print("\nLSTM model")

model_LSTM_file=f"dataset/model_{stock}_lookback_{lookBack}_LSTM.keras"
if not os.path.isfile(model_LSTM_file) or True:
    with tf.device('/CPU:0'):

        lstmm.train(lookBack, stock)
model_LSTM=keras.models.load_model(model_LSTM_file)

X_lstm,y_lstm=lstmm.create_features(df, label='Adj Close',lookBack=lookBack)
X_lstm_test=X_lstm[NTrain:].copy()
y_lstm_test=y_lstm[NTrain:].copy()

y_lstm_predict=model_LSTM.predict(X_lstm_test)#,verbose=0)

print("MSE: ",mean_squared_error(y_test.to_numpy(), y_lstm_predict))
print("MAE: ",mean_absolute_error(y_test.to_numpy(), y_lstm_predict))



"""
Average ensemble of the different models
"""

# print("\n\nAverage ensemble of the different models")
# y_predict=(y_sm_predict+y_xgb_predict+y_lstm_predict[:,0])/3
# print("MSE: ",mean_squared_error(y_test.to_numpy(), y_predict))
# print("MAE: ",mean_absolute_error(y_test.to_numpy(), y_predict))