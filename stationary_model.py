#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:40:36 2023

@author: tristan
"""

import numpy as np
import pandas as pd


"""
Stationary model
"""

def create_feature(df, label="DR",lookBack=1):
    """
        
    In this model, the daily returns are exactly 0 and we don't need an X input

    Parameters
    ----------
    df : pandas dataframe
        Contains the dataset
    label : String, optional
        Label information, redundant in the current version of the code.
        May be needed if non stationary models are considered
        The default is "DR".
    lookBack : int, optional
        Number of days to look back into.
        The default is 1.

    Returns
    -------
    y : pandas dataframe
        The daily returns of the model

    """
    
    
    if label == "DR":
        y=df["Adj Close"][lookBack:].copy()*0
    return y