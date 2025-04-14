import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

from sklearn.model_selection import ParameterGrid

import seaborn as sns


import datetime
#settings could be done as arguments
#mode=recursive #single day, range recursive?
n_splits=10
prediction_length=7   
gap=14

print(datetime.datetime.now())
try:
    df = pd.read_csv("masterdata2.csv", index_col='time', parse_dates=True)
    print("sucess")
    print(df.head())
except Exception as e:
    print(f"Error loading data file '{e}")
    exit()


#needs to also be changed in some models 
metric="mse"
def error_metric(y_true, y_pred):
        if metric=="r2":
            return r2_score(y_true, y_pred)
        elif metric=="mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric=="mse":
            return mean_squared_error(y_true, y_pred)
        else:
            return None 
        
     

print("error metric: ", metric)

def add_lags(df):
    target_map = df['level'].to_dict()

    # Add hourly lags
    #for hours in [1, 4, 8, 16, 24]:
    #    df[f'lag_{hours}h'] = (df.index - pd.Timedelta(hours, unit='h')).map(target_map)

    lag_mean_prediction_length = df['level'].rolling(window=prediction_length).mean()
    df[f'mean_{prediction_length}_lag'] = lag_mean_prediction_length.shift(prediction_length)

    #right now it is only prediciton length that adds as lag, but you could add more. ex if prediction length is 7 that 7 is the added lag, you could do so 7,8,9 is added or 7..9 
    for days in range(prediction_length,prediction_length+1):
        df[f'lag_{days}d'] = (df.index - pd.Timedelta(days, unit='d')).map(target_map)

    for days in range(365,366): #same but with year
        df[f'lag_{days}d'] = (df.index - pd.Timedelta(days, unit='d')).map(target_map)

    df = df.dropna()


    return df

df = add_lags(df)
mode="multiple"

def split():

    total_length = len(df)

    for i in range(n_splits):
        # Identify end day index for the test range
        last_test_day = total_length - (i * prediction_length*2) - ((gap)*i)
        # Identify end day index for the validation range
        last_val_day = last_test_day - prediction_length # Val ends prediction_length before test ends
        
        # Training stops prediction_length days before validation starts
        train_end_day = last_val_day - prediction_length

        # Check validity of calculated start indices
        if train_end_day < 0:
            break
        
        train_idx = np.arange(0, train_end_day+1)
        
        if(mode=="multiple"):

            # Validation set indices (range)
            val_idx = np.arange(train_end_day +1, last_val_day+1)

            # Test set indices (range)
            test_idx = np.arange( last_val_day +1, last_test_day+1 )
        
        elif(mode=="single"):
        # Validation set indices (range)
            val_idx = np.array([last_val_day])

            # Test set indices (range)
            test_idx = np.array([last_test_day ])

        yield train_idx, val_idx, test_idx
a=split()
train_idx, val_idx, test_idx= next(a)
print(train_idx, val_idx, test_idx)
train_idx, val_idx, test_idx= next(a)
print(train_idx, val_idx, test_idx)
train_idx, val_idx, test_idx= next(a)
print(train_idx, val_idx, test_idx)
print(len(df))