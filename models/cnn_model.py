import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def cnn_model(X_train, y_train, X_test, params=None):
  
 

    # Ensure numpy arrays
    X_train = X_train.values if isinstance(X_train, (pd.DataFrame, pd.Series)) else X_train
    y_train = y_train.values if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train
    X_test = X_test.values if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test

    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1: X_test = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]

    # --- Input Shape Adjustment ---
    # Keras Conv1D expects (batch_size, steps, channels/features).
    # Option 1: Treat features as steps: (batch_size, n_features, 1)
    X_train_cnn = X_train.reshape((X_train.shape[0], n_features, 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], n_features, 1))
    input_shape_cnn = (n_features, 1)

    # Option 2: Treat features as channels over 1 step: (batch_size, 1, n_features)
    # This might be less standard for Conv1D but matches RNN input shape
    # X_train_cnn = X_train.reshape((X_train.shape[0], 1, n_features))
    # X_test_cnn = X_test.reshape((X_test.shape[0], 1, n_features))
    # input_shape_cnn = (1, n_features)
    # --- End Input Shape Adjustment ---


    # Adjust kernel_size if it's larger than the sequence length (n_features here)
    current_kernel_size = min(params['kernel_size'], n_features)
    if current_kernel_size <= 0: current_kernel_size = 1

    conv_output_size = n_features - current_kernel_size + 1
    current_pool_size = params['pool_size']
    if conv_output_size < current_pool_size or conv_output_size <= 0 or current_pool_size <= 0:
        current_pool_size = 1

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape_cnn))
    model.add(layers.Conv1D(filters=params['filters'],
                           kernel_size=current_kernel_size,
                           activation=params['activation']))

    if current_pool_size > 1:
       model.add(layers.MaxPooling1D(pool_size=current_pool_size))

    model.add(layers.Flatten())
    model.add(layers.Dense(params['dense_units'], activation=params['activation']))
    model.add(layers.Dense(1))

    model.compile(optimizer=params['optimizer'],
                  loss=params['loss'])

    model.fit(X_train_cnn, y_train,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=0)

    y_pred = model.predict(X_test_cnn, verbose=params.get('verbose', 0))

    return y_pred.flatten()

# Parameter groups for hyperparameter tuning
param_groups = {
    'group1_cnn_structure': {
        'filters': [32, 64, 128, 256],
        'kernel_size': [2, 3, 5], # Will be capped by n_features
         'pool_size': [2], # Will be disabled if invalid for geometry
    },
     'group2_dense_structure': {
        'dense_units': [32, 50, 100],
     },
    'group3_activation': {
         'activation': ['relu', 'tanh'],
    },
    'group4_training': {
        'epochs': [50, 100],
        'batch_size': [32, 64],
    }
}

default_params = {
    'filters': 64,
    'kernel_size': 2,
    'activation': 'relu',
    'pool_size': 2,
    'dense_units': 50,
    'optimizer': 'adam',
    'loss': 'mse',
    'epochs': 50,
    'batch_size': 32,
    'verbose': 0
}

