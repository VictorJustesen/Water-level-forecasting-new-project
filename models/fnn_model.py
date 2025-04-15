# ffn_model.py
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def fnn_model(X_train, y_train, X_test, params=None):
   
    if params is None:
        params = {
            'units_layer1': 64,
            'activation_layer1': 'relu',
            'units_layer2': 32,
            'activation_layer2': 'relu',
            'optimizer': 'adam',
            'loss': 'mse',
            'epochs': 50,
            'batch_size': 32,
            'verbose': 0
        }

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
            layers.Dense(params.get('units_layer1', 64), activation=params.get('activation_layer1', 'relu')),
            layers.Dense(params.get('units_layer2', 32), activation=params.get('activation_layer2', 'relu')),
            layers.Dense(1) # Output layer for regression
        ]
    )

    model.compile(optimizer=params.get('optimizer', 'adam'),
                  loss=params.get('loss', 'mse'))

    model.fit(X_train, y_train,
              epochs=params.get('epochs', 50),
              batch_size=params.get('batch_size', 32),
              verbose=params.get('verbose', 0)) # Set verbose=0 to avoid excessive output during loops

    y_pred = model.predict(X_test, verbose=params.get('verbose', 0))

    return y_pred.flatten() # Flatten output to match expected shape

# Parameter groups for hyperparameter tuning
param_groups = {
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
        'optimizer': ['adam', 'rmsprop']
    }
}

