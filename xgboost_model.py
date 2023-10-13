#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:59:14 2023

@author: tristan


File to create and train an XGBoost model
"""

import pandas as pd
import matplotlib.pyplot as plt
import finance_functions as ff
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def create_features(df, label=None,lookBack=1):
    """
    (TO BE OPTIMISED)
    Create the features to be used as inputs and outputs of the xgboost model

    Parameters
    ----------
    df : pandas dataframe
        dataset on which the model will be trained and tested on.
    label : string, optional
        Name of the colomn of the dataset that will be predicted by the model.
        The default is None.
    lookBack : int, optional
        Number of days to lookback into. The default is 1.

    Returns
    -------
    X : pandas dataframe
        input of the model
    y : pandas dataframe, optional
        output of the model

    """
    
    X=pd.DataFrame()
    for i in range(len(df)-lookBack):
        date=df.index[lookBack+i]
        columns=["dayofweek","quarter","month","dayofyear","weekofyear"]
        data=[date.dayofweek,date.quarter,date.month,date.dayofyear,int(date.isocalendar().week)]

        for j in range(lookBack):
            # For day you are looking back into, add the daily returns of the adjusted closed price and exchange volume
            columns+=[f"DR-{j+1}",f"Volume -{j+1}"]
            data.append(df['Adj Close'][lookBack+i-(j+1)])
            data.append(df['Volume'][lookBack+i-(j+1)])

        temp=pd.DataFrame([data],columns=columns,index=[date])
        
        X=pd.concat([X,temp])
        
    if label:
        y=df[label][lookBack:]
        return X,y
    return X


def train(lookBack=1,stock="GOOG"):
    """
    
    Build and train the xgboost model

    Parameters
    ----------
    lookBack : int, optional
        Number of days to lookback into.
        The default is 1.
    stock : string, optional
        stock name from those available in dataset/.
        The default is "GOOG".

    Returns
    -------
    None

    """
    
    # Load the data
    df=pd.read_csv(f"dataset/{stock}.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close','Volume'],na_values=['nan'])
    df=df.dropna()
    df=df[:]
    # print(create_features(df,lookBack=1))
    # print(df)
    
    NDays=len(df)
    NTrain=int(0.8*NDays)
    # NTest=NDays-NTrain-1
    
    # M0_train=df['Adj Close'][0]
    # M0_test=df['Adj Close'][NTrain]
    # adjClose=df['Adj Close'].to_numpy()
    
    # Compute the daily returns
    df=ff.data_frame_daily_returns(df)

    # Create the features of the model, inputs X and outputs yto compare with
    X,y=create_features(df, label='Adj Close',lookBack=lookBack)
    
    #Split into training and testing sets
    X_train=X[:NTrain].copy()
    y_train=y[:NTrain].copy()
    X_test=X[NTrain:].copy()
    y_test=y[NTrain:].copy()

    # Creation of the model
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False)
    
    # plot_importance(reg, height=0.9)
    # plt.show()
    
    # print("MSE train:",mean_squared_error(y_train.to_numpy(), reg.predict(X_train)))
    # print("MAE train:",mean_absolute_error(y_train.to_numpy(), reg.predict(X_train)))
    
    # print("MSE test:",mean_squared_error(y_test.to_numpy(), reg.predict(X_test)))
    # print("MAE test:",mean_absolute_error(y_test.to_numpy(), reg.predict(X_test)))
    
    
    # AdjClose_predict=ff.daily_to_price(reg.predict(X_test),M0_test)
    # AdjClose_exact=ff.daily_to_price(y_test,M0_test)
    
    # plt.plot(AdjClose_exact)
    # plt.plot(AdjClose_predict)
    # plt.show()
    


    
    # Save the model
    reg.save_model(f"dataset/model_{stock}_lookback_{lookBack}_xgboost.json")