# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:00:08 2023

@author: tmade


Ensemble of functions relevant for financial data manipulation
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

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
    dr_pf=df.pct_change(1)
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