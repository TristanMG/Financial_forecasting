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
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import finance_functions as ff

def model_LSTM(lookBack=1,NNN=10,N_layers=2,DO=0,lr=1e-2):
    """
    Function to build the neural network made of LSTM nodes 

    Parameters
    ----------
    lookBack : int, optional
        Number of days to lookback into. The default is 1.
    NNN : int, optional
        Number of nodes per layer. The default is 10.
    N_layers : int, optional
        Number of layers. The default is 2.
    DO : float, optional
        Dropout rate. The default is 0.
    lr : float, optional
        learning rate. The default is 1e-2.

    Returns
    -------
    model : keras.model
       LSTM network model

    """
    
    model=Sequential([layers.LSTM(NNN,input_shape=(2*lookBack, 1), return_sequences=True,activation="tanh"),
    layers.Dropout(DO)])
    
    for i in range(N_layers-2):
        model.add(layers.LSTM(NNN, return_sequences=True,activation="tanh"))
        model.add(layers.Dropout(DO))    
    model.add(layers.LSTM(NNN,activation="tanh"))
    model.add(layers.Dropout(DO))  
    model.add(layers.Dense(1,activation="linear"))
    
    model.compile(loss='mean_squared_error',  optimizer=Adam(learning_rate=lr),metrics = ['mse', 'mae'])
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


def train(folder="SP500",stock="GOOG",date_test_string="2014-12-31",date_validation_string="2018-12-31",lookBack=1,options=None):
    """
    Build, train and save the LSTM model

    Parameters
    ----------
    folder : string, optional
        Folder in which the stock is. The default is "SP500_cleaned".
    stock : string, optional
        Name of the stock. The default is "GOOG".
    options : dict, optional
        Parameters of the ML architecture, learning process and shape of the input data.
        The default is None.

    Returns
    -------
    None.

    """
    
    if options == None:
        NNN=10
        layers=2
        dropout=0.2
        learning_rate=0.01
    else:
        NNN=options["NNN"]
        layers=options["layers"]
        dropout=options["dropout"]
        learning_rate=options["learning_rate"]
    
    date_test=pd.Timestamp(date_test_string)
    date_validation=pd.Timestamp(date_validation_string)
    
    # Load the data
    X_train,y_train=ff.create_features(stock_l=[stock],market=folder,lookBack=lookBack,label="Adj Close",date_test=date_test,date_validation=date_validation,set_type="train")
    X_test,y_test=ff.create_features(stock_l=[stock],market=folder,lookBack=lookBack,label="Adj Close",date_test=date_test,date_validation=date_validation,set_type="test")
    
    
    # Create the model
    model=model_LSTM(lookBack=lookBack,NNN=NNN,N_layers=layers,DO=dropout,lr=learning_rate)
    
    #Training parameters
    # early_stopping = EarlyStopping(
    #     min_delta=0.0000001, # minimium amount of change to count as an improvement
    #     patience=20, # how many epochs to wait before stopping
    #     restore_best_weights=True,
    # )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=20,min_delta=0.0000001,restore_best_weights=True)
    
    # Train the model
    # history=model.fit(X_train,y_train, epochs=1000, batch_size=4000,verbose=0, validation_data=(X_test,y_test),callbacks=[early_stopping],shuffle=True)
    history=model.fit(X_train,y_train, epochs=60, batch_size=200000,verbose=0, validation_data=(X_test,y_test), callbacks=[stop_early])
    
    # model_loss(history)
    
    # y_predict=model.predict(X_test)
    
    # print("MSE: ",mean_squared_error(y_test.to_numpy(), y_predict))
    # print("MAE: ",mean_absolute_error(y_test.to_numpy(), y_predict))
    
    #Save the model
    if stock == "all":
        model.save(f"{folder}_prediction/model_{stock}_LSTM_lookback_{lookBack}_date_{date_test_string}.keras")
    else:
        model.save(f"{folder}_prediction/{stock}_LSTM_lookback_{lookBack}_date_{date_test_string}.keras")

