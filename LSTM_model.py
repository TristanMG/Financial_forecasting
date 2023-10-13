#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:17:49 2023

@author: tristan

File to create and train a neural network based on LSTM nodes
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import finance_functions as ff


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
        columns=[]
        data=[]
        for j in range(lookBack):
            # For day you are looking back into, add the daily returns of the adjusted closed price and exchange volume
            columns+=[f"DR-{j+1}",f"Volume -{j+1}"]
            data.append(df['Adj Close'][lookBack+i-(j+1)])
            data.append(df['Volume'][lookBack+i-(j+1)])
        # print(columns,data)
        temp=pd.DataFrame([data],columns=columns,index=[date])
        
        X=pd.concat([X,temp])
        
    if label:
        y=df[label][lookBack:]
        return X,y
    return X



def model_LSTM(look_back=1):
    """
    
    Function to build the neural network made of LSTM nodes 
    Parameters
    ----------
    lookBack : int, optional
        Number of days to lookback into. The default is 1.

    Returns
    -------
    model : keras.model
       Neural network model

    """
    # Number of nodes per layer
    NNN=10
    #Fraction of dropouts
    DO=0.3
    model=Sequential([        
        layers.LSTM(NNN,input_shape=(2*look_back, 1), return_sequences=True),
        layers.Dropout(DO),        
        layers.LSTM(NNN),
        layers.Dropout(DO),       
        layers.Dense(1,activation="linear"),
        ])
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])
    return model

def model_loss(history):
    """
    
    Displays the training set and testing set losses of the model during training

    Parameters
    ----------
    history : fit of keras model
        output of model.fit().

    Returns
    -------
    None.

    """
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();


def train(lookBack=1,stock="GOOG"):
    """
    
    Build and train the LSTM model

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
    
    NDays=len(df)
    NTrain=int(0.8*NDays)
    
    # Compute the daily returns
    df=ff.data_frame_daily_returns(df)
    
    #Create the features to train the model
    X,y=create_features(df, label='Adj Close',lookBack=lookBack)
    
    #Split into training and testing sets
    X_train=X[:NTrain].copy()
    y_train=y[:NTrain].copy()
    X_test=X[NTrain:].copy()
    y_test=y[NTrain:].copy()
    
    # Create the model
    model=model_LSTM(lookBack)
    
    #Training parameters
    early_stopping = EarlyStopping(
        min_delta=0.0000001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    
    
    # Train the model
    history=model.fit(X_train,y_train, epochs=1000, batch_size=4000,verbose=0, validation_data=(X_test,y_test),callbacks=[early_stopping],shuffle=True)
    
    # model_loss(history)
    
    # y_predict=model.predict(X_test)
    
    # print("MSE: ",mean_squared_error(y_test.to_numpy(), y_predict))
    # print("MAE: ",mean_absolute_error(y_test.to_numpy(), y_predict))
    
    #Save the model
    model.save(f"dataset/model_{stock}_lookback_{lookBack}_LSTM.keras")

