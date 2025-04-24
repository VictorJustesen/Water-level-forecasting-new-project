import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def fnn_model(X_train, y_train, X_test, params=None):
   
     

    # Ensure numpy arrays
    X_train = X_train.values if isinstance(X_train, (pd.DataFrame, pd.Series)) else X_train
    y_train = y_train.values if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train
    X_test = X_test.values if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test

    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1: X_test = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(params['units_layer1'], activation=params['activation_layer1']),
            layers.Dropout(params['dropout1']),
            layers.Dense(params['units_layer2'], activation=params['activation_layer2']),
            layers.Dropout(params['dropout2']),
            layers.Dense(1)
        ]
    )

    model.compile(optimizer=params['optimizer'],
                  loss=params['loss'])

    model.fit(X_train, y_train,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=0)

    y_pred = model.predict(X_test, verbose=0)

    return y_pred.flatten() # Flatten output to match expected shape

# Parameter groups for hyperparameter tuning
param_groups = {
    'dropout_rates': {
    'dropout1': [0.0, 0.1, 0.2, 0.3],
    'dropout2': [0.0, 0.1, 0.2]
},
    'group1_structure': {
        'units_layer1': [32, 64, 128],
        'units_layer2': [16, 32, 64],
    },
    'group2_activation': {
         'activation_layer1': ['relu', 'tanh'],
         'activation_layer2': ['relu', 'tanh'],
    },
    'group3_training': {
        'epochs': [50, 100],
        'batch_size': [32, 64],
    }
}

default_params = {
    'units_layer1': 64,
    'activation_layer1': 'relu',
    'units_layer2': 32,
    'activation_layer2': 'relu',
    'optimizer': 'adam',
    'loss': 'mse',
    'epochs': 50,
    'batch_size': 32,
    'verbose': 0,
    'dropout1': 0.0,
    'dropout2': 0.0
}