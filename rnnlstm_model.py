# rnn_model.py
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def rnnlstm_model(X_train, y_train, X_test, params=None):
    """
    Trains and predicts using a Recurrent Neural Network (RNN), specifically LSTM.
    Assumes input features represent the state at a single time step.
    """
    if params is None:
        params = {
            'units': 50,
            'activation': 'relu',
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

    # Reshape input for RNN: (n_samples, n_timesteps, n_features)
    # We treat the provided features as a single time step
    n_features = X_train.shape[1]
    X_train_rnn = X_train.reshape((X_train.shape[0], 1, n_features))
    X_test_rnn = X_test.reshape((X_test.shape[0], 1, n_features))

    model = keras.Sequential(
        [
            layers.Input(shape=(1, n_features)), # 1 time step
            # Using LSTM layer
            layers.LSTM(params.get('units', 50), activation=params.get('activation', 'relu')),
            layers.Dense(1) # Output layer for regression
        ]
    )

    model.compile(optimizer=params.get('optimizer', 'adam'),
                  loss=params.get('loss', 'mse'))

    model.fit(X_train_rnn, y_train,
              epochs=params.get('epochs', 50),
              batch_size=params.get('batch_size', 32),
              verbose=params.get('verbose', 0))

    y_pred = model.predict(X_test_rnn, verbose=params.get('verbose', 0))

    return y_pred.flatten()

# Parameter groups for hyperparameter tuning
param_groups = {
    'group1_structure': {
        'units': [32, 50, 100],
    },
    'group2_activation': {
         'activation': ['relu', 'tanh'],
    },
    'group3_training': {
        'epochs': [50, 100],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'rmsprop']
    }
}

model_type = 'multivariate'