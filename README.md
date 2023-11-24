# Financial_forecasting
Implementation of an ensemble machine learning method to predict the daily change of the closed price of a stock.

Stock prices follow complex trajectories making them virtually impossible to predict. In their simplest model, they have Brownian motion-like properties with Markovian properties, i.e. the best guess of the price of tomorrow is the price of today.

I implemented and combined four different LSTM networks built using [keras](https://www.tensorflow.org/guide/keras) to try to predict the evolution of a stock's price from one day to an other.
These models use information about the previous daily returns and exchange volumes. They differ by the number of previous days they use as inputs, and whether they have been trained on data from a single company (load_LSTM.py) or every company in the market (load_LSTM_all.py).
The hyperparameters of most models were found via Bayesian optimisation.
For each company, the training set was financial data up to the end of 2014 and the testing set was financial data between 2015 and 2018. The ensemble learning is a regression of the different models, trained with financial data up to 2018.

170 companies from FTSE250 and 439 companies from SP500 were considered in this study. After training, the daily evolution of their price was predicted up to November 2023. These predicted price evolutions were used to build optimised portfolios on a daily basis. The evolution of the value of these portfolio made of companies of the London and american market is reported, alongside the market indices as benchmarks, in the following two images.


<img src=https://github.com/TristanMG/Financial_forecasting/blob/main/images/Optimised_portfolio_vs_index_FTSE.png width="45%"> <img src=https://github.com/TristanMG/Financial_forecasting/blob/main/images/Optimised_portfolio_vs_index_SP500.png width="45%">
