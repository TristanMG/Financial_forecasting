# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:00:08 2023

@author: tmade


Ensemble of functions relevant for financial data manipulation
"""

import pandas as pd
import numpy as np
import glob

# import matplotlib.pyplot as plt

def create_features_old(stock_l, market="FTSE_cleaned", label=None,lookBack=1,test=False,date=pd.Timestamp("2014-12-31")):
    """
    
    Parameters
    ----------
    stock : string
        Name of the stock, must be present in the folder {market}
    market : string, optional
        Name of the market. The default is "FTSE".
    label : string, optional
        Quantity to predict, if provided. The default is None.
    lookBack : int, optional
        Number of days to lookback into. The default is 1.
    test : bool, optional
        If the daya used will be part of the testing or training dataset.
        The default is False.
    date : pandas timestamp, optional
        date up to (excluding) (for training set) and from to (for testing set).
        The default is pd.Timestamp("2014-12-31").

    Returns
    -------
    X : array
        Name of the stock, must be present in the folder {market}
    y : string, optional
        Name of the market. The default is "FTSE".
    """
    
    if len(stock_l)==1 and stock_l[0] == "all":
        temp_l=glob.glob(market+"/*.csv")
        stock_l=[]
        for temp in temp_l:
            stock_l.append(temp.split("\\")[1][:-4])
    
    X_l,y_l=[],[]
    
    for stock in stock_l:
        
        
        df=pd.read_csv(f"{market}/{stock}.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close','Volume'],na_values=['nan'])
        # print(df)
        index=np.where(df.index==date)[0][0]
        # print(index)
        if test:
            df=df.iloc[index-1-lookBack:]
            if label != None:
                y=df[label][index-1:]
            
        else:
            if label != None:
                y=df[label][lookBack:index]
            df=df.iloc[:index]
            
        df=df.dropna()
    
        #Compute the daily returns of both the Adjusted close price and the exchange volume
        df=data_frame_daily_returns(df)
        
        if label != None:
            y=data_frame_daily_returns(y).to_numpy()
        
        # print(df)
        data=df.to_numpy()
        
        #Replace the apparent infinite evolution of the volume (due to days recorded with no volume exchanged) by an arbitrarily large number
        data[data == np.inf] = 5
        
        X=data[lookBack:]
        
        for i in range(lookBack):
            temp=data[lookBack-(i+1):-(i+1)]
            X=np.concatenate((X,temp),axis=1)
            
        X_l.append(X[:,2:])
        if label != None:
            y_l.append(y)
            
        
    if label != None:
        y=np.array([])
        for temp in y_l:
            # print(temp)
            y=np.concatenate((y,temp),axis=0)
        
    X=np.zeros((1,X_l[0].shape[1]))
    for temp in X_l:
        X=np.concatenate((X,temp),axis=0)
        
    if label != None:
        return X[1:],y
    
    return X[1:]





def create_features(stock_l, market="FTSE_cleaned", label=None,lookBack=1,set_type=False,date_test=pd.Timestamp("2014-12-31"),date_validation=pd.Timestamp("2018-12-31")):
    """
    
    Parameters
    ----------
    stock : string
        Name of the stock, must be present in the folder {market}
    market : string, optional
        Name of the market. The default is "FTSE".
    label : string, optional
        Quantity to predict, if provided. The default is None.
    lookBack : int, optional
        Number of days to lookback into. The default is 1.
    test : bool, optional
        If the daya used will be part of the testing or training dataset.
        The default is False.
    date : pandas timestamp, optional
        date up to (excluding) (for training set) and from to (for testing set).
        The default is pd.Timestamp("2014-12-31").

    Returns
    -------
    X : array
        Name of the stock, must be present in the folder {market}
    y : string, optional
        Name of the market. The default is "FTSE".

    """
    
    if len(stock_l)==1 and stock_l[0] == "all":
        temp_l=glob.glob(market+"/*.csv")
        stock_l=[]
        for temp in temp_l:
            stock_l.append(temp.split("\\")[1][:-4])
    
    X_l,y_l=[],[]
    
    for stock in stock_l:
        
        
        df=pd.read_csv(f"{market}/{stock}.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close','Volume'],na_values=['nan'])
        # print(df)
        index_test=np.where(df.index==date_test)[0][0]
        index_validation=np.where(df.index==date_validation)[0][0]
        
        if set_type=="test":
            # label=None
            if label != None:
                y=df[label][index_test-1:index_validation]
            df=df.iloc[index_test-1-lookBack:index_validation]
            
        elif set_type=="train":
            if label != None:
                y=df[label][lookBack:index_test]
            df=df.iloc[:index_test]
            
        elif set_type=="validation":
            if label != None:
                y=df[label][index_validation-1:]
            df=df.iloc[index_validation-1-lookBack:]
        else:
            return
            
            
        df=df.dropna()
    
        #Compute the daily returns of both the Adjusted close price and the exchange volume
        df=data_frame_daily_returns(df)
        if label != None:
            y=data_frame_daily_returns(y).to_numpy()
        
        # print(df)
        data=df.to_numpy()
        
        #Replace the apparent infinite evolution of the volume (due to days recorded with no volume exchanged) by an arbitrarily large number
        data[data == np.inf] = 5
        
        X=data[lookBack:]
        
        for i in range(lookBack):
            temp=data[lookBack-(i+1):-(i+1)]
            X=np.concatenate((X,temp),axis=1)
            
        X_l.append(X[:,2:])
        if label != None:
            y_l.append(y)
            
        
    if label != None:
        y=np.array([])
        for temp in y_l:
            # print(temp)
            y=np.concatenate((y,temp),axis=0)
        
    X=np.zeros((1,X_l[0].shape[1]))
    for temp in X_l:
        X=np.concatenate((X,temp),axis=0)
        
    if label != None:
        return X[1:],y
    
    return X[1:]



def daily_to_price(t,m0):
    """
    Assuming a series of daily returns (t) and an original
    stock price of value (m0), compute the evolution of the stock price

    Parameters
    ----------
    t : array
        daily returns in terms of the indexed time
    m0 : float
        initial stock price of value

    Returns
    -------
    temp : array
        Stock price, indexed by time

    """
    temp=np.zeros(len(t)+1)
    temp[0]=m0
    for i in range(len(t)):
        temp[i+1]=temp[i]*(1+t[i])
    return temp



def data_frame_daily_returns(df):
    """
    Compute the daily returns of each column of a dataframe

    Parameters
    ----------
    df : pandas dataframe
        dataset

    Returns
    -------
    dr_pf : pandas dataframe
        daily returns of each column of the dataset

    """
    # dr_pf=df.pct_change(1)
    dr_pf=df.ffill().pct_change(1)
    dr_pf=dr_pf.fillna(0)[1:]
    return dr_pf



def portfolio_statistics(pf):
    """
    Compute statistical parameters of a portfolio

    Parameters
    ----------
    pf : pandas dataframe
        portfolio of stocks

    Returns
    -------
    CR : float
        percentage change of an asset
    avg_daily_returns : float
        mean of the daily returns.
    std_daily_returns : float
        standard deviation of the daily returns.
    sharpe_ratio : float
        sharpe ratio.

    """
    CR=pf[-1]/pf[0]-1
    avg_daily_returns=data_frame_daily_returns(pf).mean()
    std_daily_returns=data_frame_daily_returns(pf).std()
    sharpe_ratio=np.sqrt(252)*avg_daily_returns/std_daily_returns
    
    return CR,avg_daily_returns,std_daily_returns,sharpe_ratio

def sharpe_ratio(w,args):
    """
    Sharpe ratio computed for a portfolio made of the assets in quantities given by w

    Parameters
    ----------
    w : numpy array
        weight of the assets of the portfolio
    args : dict
        Additional parameters to compute the sharpe ratio:
            mu_l: average fo the daily returns of each asset
            cov: covariance matrix of the daily returns of the assets
            Rf: the risk-free rate
            T: the time horizon

    Returns
    -------
    float
        The sharpe ratio

    """
    mu_l=args["mu_l"]
    cov=args["cov"]
    Rf=args["Rf"]
    T=args["T"]
    return_p=mu_l.dot(w)
    covariance_p=cov.dot(w).dot(w)
    return (return_p-Rf)/np.sqrt(covariance_p/T)