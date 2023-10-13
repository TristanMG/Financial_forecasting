# Financial_forecasting
Implementation of an ensemble machine learning method to predict the daily change of the closed price of a stock.

Stock prices follow complex trajectories making them virtually impossible to predict. In their simplest model, they follow Brownian motion-like properties with Markovian properties, i.e. the best guess of the price of tomorrow is the price of today.


I implemented, and combined three different models to try to predict the evolution of a stock's price from one day to an other.

The first and the simplest model is the assumption that the price remains constant from one day to the other. Its description is in stationary_model.py.

The second model is a random forest model using [xgboost](https://github.com/dmlc/xgboost), using information about the date and the previous daily returns and exchange volumes.  Its description is in xgboost_model.py.

The third model is a LSTM neural network using [keras](https://www.tensorflow.org/guide/keras), using information about the previous daily returns and exchange volumes.  Its description is in LSTM_model.py.


The handling and combination of the models are in load_models.py
