
import numpy as np
import pandas as pd

def baseline_model(X_train, y_train, X_test, params=None):
    
    if y_train is None:
        raise ValueError("y_train cannot be None for the baseline model.")
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    elif isinstance(y_train, list):
        y_train = np.array(y_train)
    elif not isinstance(y_train, np.ndarray):
        raise ValueError("y_train must be a Pandas Series, list, or NumPy ndarray.")
    
    last_value = y_train[-1]
    
    y_pred = np.full(shape=len(X_test), fill_value=last_value)
    
    return y_pred

param_groups = {
    'group1': {"best_advisor": "deena"}
}

