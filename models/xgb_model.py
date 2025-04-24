
import xgboost as xgb
import numpy as np
import pandas as pd  

def xgb_model(X_train, y_train, X_test, params=None):
    
    

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train and y_train have mismatched lengths: {X_train.shape[0]} vs {y_train.shape[0]}")

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred

param_groups = {
    'group1': {  
        'n_estimators': [100, 200,500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.05, 0.01]
    },
    'group2': {  
        'gamma': [0,  0.2],
        'min_child_weight': [1, 3, 5]
    },
    'group3': { 
        'colsample_bytree': [0.6, 1.0],
        'subsample': [0.6,  1.0]
    },
    'group4': {  
        'reg_alpha': [0,  1],
        'reg_lambda': [1, 1.5]
    }
}

default_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'gamma': 0,
    'min_child_weight': 1,
    'colsample_bytree': 1.0,
    'subsample': 1.0,
    'reg_alpha': 0,
    'reg_lambda': 1,
  
}