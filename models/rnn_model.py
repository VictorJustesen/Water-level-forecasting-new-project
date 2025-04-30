import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from .create_sequences import create_sequences

def rnn_model(X_train, y_train, X_test, params=None):
    #print(f'training with x_train: {X_train},y_train: {y_train}, x_test: {X_test}, params: {params}')
    

    # Ensure numpy arrays
    X_train = X_train.values if isinstance(X_train, (pd.DataFrame, pd.Series)) else X_train
    y_train = y_train.values if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train
    X_test = X_test.values if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test

    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1: X_test = X_test.reshape(-1, 1)

    # Reshape input for RNN: (n_samples, n_timesteps, n_features)
    # We treat the provided features as a single time step
    n_features = X_train.shape[1]
    X_train_seq = X_train.reshape((X_train.shape[0], 1, n_features))
    X_test_seq = X_test.reshape((X_test.shape[0], 1, n_features))

    seq_length = params['seq_length']
    
    # Create training sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    
    # Handle test sequences based on prediction mode
    if X_test.shape[0] == 1:  # Single prediction
        # For single prediction, use the last seq_length observations from training
        X_test_seq = np.array([X_train[-seq_length:]])
    else:  # Multiple predictions
        test_size = X_test.shape[0]
        if test_size < seq_length:
            # If test data is smaller than sequence length, combine with end of training data
            combined = np.vstack((X_train[-seq_length+test_size:], X_test))
            X_test_seq = np.array([combined[-seq_length:]])
        else: # test_size >= seq_length
            required_train_history = X_train[-seq_length:]

            combined_data = np.vstack((required_train_history, X_test))

            X_test_seq = []
            for i in range(test_size):
                start_index = i
                end_index = i + seq_length
                X_test_seq.append(combined_data[start_index:end_index])

            X_test_seq = np.array(X_test_seq)
    
    # Number of features
    n_features = X_train.shape[1]

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length, n_features)), 
            # Using simplernn layer
            layers.SimpleRNN(params['units'], activation=params['activation'], dropout=params['dropout']),
            layers.Dense(1)
        ]
    )

    model.compile(optimizer=params['optimizer'],
                  loss=params['loss'])

    model.fit(X_train_seq, y_train_seq,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=0,
              shuffle=False )

    y_pred = model.predict(X_test_seq,verbose=0)

    return y_pred.flatten()

# Parameter groups for hyperparameter tuning
param_groups = {
       'dropout': {
    'dropout': [0.0, 0.1, 0.2, 0.3],
},
    'group1_structure': {
        'units': [32, 50, 100],
    },
    'group2_activation': {
         'activation': ['relu', 'tanh'],
    },
    'group3_training': {
        'epochs': [50, 100],
        'batch_size': [32, 64],
    },
    'sequence_length': {
        'seq_length': [3, 7, 14]
    },
}

default_params = {
    'units': 50,
    'activation': 'relu',
    'optimizer': 'adam',
    'loss': 'mse',
    'epochs': 50,
    'batch_size': 32,
    'dropout': 0.0,
    'seq_length': 7,
}
