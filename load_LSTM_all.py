# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:44:28 2023

@author: tmade
"""

import os.path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
import keras
import LSTM_model as lstmm
import finance_functions as ff
from glob import glob as globi
import json

market="FTSE"
# market="SP500"

date_test_string="2014-12-31"
date_validation_string="2018-12-31"
lookBack=1

stock="all"


path_model=f"{market}_prediction/model_all_LSTM_lookback_{lookBack}_date_{date_test_string}.keras"

if not os.path.exists(path_model):
    path_options=f"{market}\\all_LSTM_lookBack_{lookBack}.txt"
    if os.path.exists(path_options):
        # print(stock)
        
        with open(path_options) as json_file:
            options = json.load(json_file)
    else:
        options=None
            
    lstmm.train(folder=market,stock=stock,date_test_string=date_test_string,date_validation_string=date_validation_string,lookBack=lookBack,options=options)
            

model_LSTM=keras.models.load_model(path_model)



stock_list=globi(market+"/*.csv")
stock_list.sort()

# print(stock_list)


for i,stock in enumerate(stock_list[:]):
    stock=stock.split("\\")[1]
    print(stock,f"{i+1}/{len(stock_list)}")
    
    
    path_validation_prediction=f"{market}_prediction\\"+stock[:-4]+f"_all_LSTM_lookBack_{lookBack}_date_{date_validation_string}.txt"
    path_test_prediction=f"{market}_prediction\\"+stock[:-4]+f"_all_LSTM_lookBack_{lookBack}_test_date_{date_test_string}_date_{date_validation_string}.txt"
    path_reference=f"{market}_prediction\\"+stock[:-4]+f"_date_{date_validation_string}_reference.txt"
    path_reference_test=f"{market}_prediction\\"+stock[:-4]+f"_test_date_{date_test_string}_date_{date_validation_string}_reference.txt"
    
    
    if not (os.path.exists(path_validation_prediction) and  os.path.exists(path_test_prediction) and os.path.exists(path_reference_test)) :
        
       
        
        
        X_validation,y_validation=ff.create_features(stock_l=[stock[:-4]],market=market,lookBack=lookBack,label="Adj Close",date_test=pd.Timestamp(date_test_string),date_validation=pd.Timestamp(date_validation_string),set_type="validation")
        
        y_predict=model_LSTM.predict(X_validation)

        print("MSE: ",mean_squared_error(y_validation, y_predict))
        print("MSE stationarry: ",mean_squared_error(y_validation, y_predict*0))
        print("MAE: ",mean_absolute_error(y_validation, y_predict))
        print("MAE stationnary: ",mean_absolute_error(y_validation, y_predict*0))
        
        np.savetxt(path_validation_prediction,y_predict[:,0])
        np.savetxt(path_reference,y_validation)
        
        X_test,y_test=ff.create_features(stock_l=[stock[:-4]],market=market,lookBack=lookBack,label="Adj Close",date_test=pd.Timestamp(date_test_string),date_validation=pd.Timestamp(date_validation_string),set_type="test")
        np.savetxt(path_test_prediction,model_LSTM.predict(X_test)[:,0])
        np.savetxt(path_reference_test,y_test)
        print()
            