import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

from sklearn.model_selection import ParameterGrid

import seaborn as sns

import linear_model
import fnn_model
import rnn_model
import cnn_model
import baseline_model
import rf_model
import xgb_model
import rnnlstm_model
import datetime
#settings could be done as arguments
mode=recursive #single day, range recursive?
n_splits=10
prediction_length=7   

print(datetime.datetime.now())
try:
    df = pd.read_csv("masterdata2.csv", index_col='time', parse_dates=True)
    print("sucess")
    print(df.head())
except Exception as e:
    print(f"Error loading data file '{e}")
    exit()

#models, you can out comment
models = [
    'linear_model',
    #'rf_model',
    #'xgb_model',
    'fnn_model',
    'rnn_model',
    'cnn_model',
    'rnnlstm_model',
    'baseline_model',
]

imported_models = {}
model_params_groups = {}
model_types = {}
# get attributes in maps
for model_name in models:
    module = globals()[model_name] 
    imported_models[model_name] = getattr(module, model_name)
    model_params_groups[model_name] = getattr(module, 'param_groups')
    model_types[model_name] = getattr(module, 'model_type')

print("running models: ", models)
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

def split(n_splits=n_splits ,
          k=prediction_length,       
          gap=14,
          df=df):

    total_length = len(df)
    
    for i in range(n_splits):
        # Identify test day
        test_day = total_length - i * gap - 1  # one day at the end, then move backward by interval each split
        
        # Validation day is k days before the test day
        val_day = test_day - k
        
        # Check validity of indices
        if val_day < 0 or test_day >= total_length:
            break
        
        # Training set: from the start up to k day before validation
        train_idx = np.arange(0, val_day -k-1 )
        
        # Validation set: single day
        val_idx = np.array([val_day])
        
        # Test set: single day
        test_idx = np.array([test_day])

        
        yield train_idx, val_idx, test_idx

# # Define test size and validation size
# display_splits = min(5, n_splits)  # Number of splits to display

# # Plotting
# fig, axs = plt.subplots(display_splits, 1, figsize=(15, 5 * display_splits))

# count = 0
# for train_idx, val_idx, test_idx in split():  
#     if count >= display_splits:  
#         break

#     train = df.iloc[train_idx]
#     val = df.iloc[val_idx]
#     test = df.iloc[test_idx]

#     axs[count].plot(train.index, train["level"], label='Train Data', color='blue')
#     axs[count].plot(val.index, val["level"], label='Validation Data', color='green')
#     axs[count].plot(test.index, test["level"], label='Test Data', color='red')
#     axs[count].set_title(f"Train/Validation/Test Split {count + 1}")
#     axs[count].legend()

#     count += 1  

# plt.tight_layout() 
# plt.show()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
print(df.columns)



print("feature selection")
selected_features_dict = {}

# Sequential Feature Selection Loop
for model_name in models:
    print(f"\nPerforming feature selection for {model_name}")
    print(datetime.datetime.now())

    #i had univariate at some point but now there is just multivariate and baseline 
    model_type = model_types[model_name]
    model_func = imported_models[model_name]
    # sequential feature selection, we try to add features to see if the model is better
    if model_type == 'multivariate':
        #the ones we select
        selected_features = []
        remaining_features = [col for col in df.columns if col != 'level']
        best_score = float('inf')
        # try to add one we havent added yet
        while remaining_features:
            scores = {}
            #we do it for all features and take the best one if better than the current model
            for feature in remaining_features:
                current_features = selected_features + [feature]
                errors = []
                for train_idx, val_idx, test_idx in split():
                    train_df = df.iloc[train_idx]
                    val_df = df.iloc[val_idx]
                    
                    X_train = train_df[current_features]
                    y_train = train_df['level']
                    X_val = val_df[current_features]
                    y_val = val_df['level']
                    
                    if isinstance(X_train, pd.Series):
                        X_train = X_train.to_frame()
                    if isinstance(X_val, pd.Series):
                        X_val = X_val.to_frame()
                    
                    try:
                        # train the model with the new feature
                        y_pred = model_func(X_train, y_train, X_val, params={})
                        error = error_metric(y_val, y_pred)
                    except Exception as e:
                        print(f"Error with feature '{feature}' in model '{model_name}': {e}")
                        error = float('inf')
                    
                    errors.append(error)
                
                mean_error = np.mean(errors)
                scores[feature] = mean_error
                print(f"Tested feature '{feature}': Mean {metric} = {mean_error}")
            #take the best feature
            best_feature, best_feature_score = min(scores.items(), key=lambda x: x[1])
            #only take it if better than current model score
            if best_feature_score < best_score:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_score = best_feature_score
                print(f"Selected feature '{best_feature}' with improvement to {metric} = {best_feature_score}")
            else:
                print("No further improvement, stopping feature selection.")
                break
        
        print(f"Selected features for '{model_name}': {selected_features}")
        selected_features_dict[model_name] = selected_features
    else:
        selected_features = ['level']
        selected_features_dict[model_name] = selected_features
        print(f"Univariate model selected. Using 'level' as the only feature for '{model_name}'.")


#hyperperameter tuning 
# Initialize dictionaries to store best parameters and errors
best_params_dict = {}
best_errors_dict = {}
for model_name in models:
    if model_name == "baseline_model":
        continue
    
    print(f"\nStarting hyperparameter tuning for '{model_name}'")
    print(datetime.datetime.now())

    model_type = model_types[model_name]
    model_func = imported_models[model_name]
    param_groups = model_params_groups[model_name]
    selected_features = selected_features_dict.get(model_name, ['level'])
    
    best_error = float('inf')
    best_params = {}
    # so we this is this grid search but witihin groups, so we try all combinations of the parameters in a group and then we go over the groups,
    # its pretty close to sequential feature selection
    for group_name, group_params in param_groups.items():
        print(f"Optimizing parameter group '{group_name}' for '{model_name}'")
        param_grid = list(ParameterGrid(group_params))
        
        for params in param_grid:
            current_params = best_params.copy()
            current_params.update(params)
            
            errors = []
            for train_idx, val_idx, test_idx in split():
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                y_val = val_df['level'].values
                
                
                if model_type == 'multivariate':
                    X_train = train_df[selected_features]
                    y_train = train_df['level']
                    X_val = val_df[selected_features]
                    
                    
                    # Ensure DataFrames
                    if isinstance(X_train, pd.Series):
                        X_train = X_train.to_frame()
                    if isinstance(X_val, pd.Series):
                        X_val = X_val.to_frame()
                    #train the model
                    try:
                        y_pred = model_func(X_train, y_train, X_val, current_params)
                        error = error_metric(y_val, y_pred)
                    except Exception as e:
                        print(f"Error during parameter tuning for '{model_name}' with params {params}: {e}")
                        error = float('inf')
                    
                    errors.append(error)
                else:
                    pass 
                
            mean_error = np.mean(errors)
            
            print(f"Tested params {params}: Mean {metric} = {mean_error}")
            #only change if better than current model
            if mean_error < best_error:
                best_error = mean_error
                best_params = current_params.copy()
                print(f"New best params for '{model_name}': {best_params} with Mean {metric} = {best_error}")
    
    if best_params:
        best_params_dict[model_name] = best_params
        best_errors_dict[model_name] = best_error
        print(f"Best parameters for '{model_name}': {best_params}")
        print(f"Best mean error for '{model_name}': {best_error}")
    else:
        print(f"No valid parameter set found for '{model_name}'.")


#final models, run the next to see test result and the besdt model 
for model_name in best_params_dict.keys():
    model_func = imported_models.get(model_name)
    
    best_params = best_params_dict[model_name]
    model_error = best_errors_dict.get(model_name)
    
    selected_features = selected_features_dict.get(model_name, ['level'])
    
    print(f"Model: {model_name}")
    print(f"  Selected Features: {selected_features}")
    print(f"  Best Parameters: {best_params}")
    print(f"  Best Error: {model_error}")



final_results = []

# Evaluate results for all models 
for model_name in models:
    print(f"\nEvaluating final model for '{model_name}'")
    
    if model_name != "baseline_model":
        # Get function and parameters for other models
        model_func = imported_models[model_name]
        best_params = best_params_dict.get(model_name, {})
        selected_features = selected_features_dict.get(model_name, ['level'])
    else:
        # Handle Baseline Model
        best_params = None
        selected_features = None
    
    errors = []
    
    for i, (train_idx, val_idx, test_idx) in enumerate(list(split())):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
        y_test = test_df['level']
        #predecting
        if model_name == "baseline_model":
            y_pred = test_df['lag_7d'].values
        else:
            # Combine training and validation data for final training
            X_full_train = pd.concat([train_df[selected_features], val_df[selected_features]])
            y_full_train = pd.concat([train_df['level'], val_df['level']])
            X_test = test_df[selected_features]
            y_test = test_df['level'].values
            # Ensure DataFrames
            if isinstance(X_full_train, pd.Series):
                X_full_train = X_full_train.to_frame()
            if isinstance(X_test, pd.Series):
                X_test = X_test.to_frame()
            
            y_pred = model_func(X_full_train, y_full_train, X_test, best_params)
            
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        error = error_metric(y_test, y_pred)
        errors.append(error)
    
    mean_error = np.mean(errors)
    best_errors_dict[model_name] = mean_error
    print(f"Final Mean {metric} for '{model_name}': {mean_error}")
    final_results.append((model_name, mean_error))
    

# Find the best model
best_model_name, best_model_error = min(final_results, key=lambda x: x[1])

print("\nModel Performance Comparison:")
for model_name, error in final_results:
    print(f"Model: {model_name}, Mean {metric}: {error}")

print(f"\nBest Model: {best_model_name} with Mean {metric}: {best_model_error}")
print(f"Baseline Model Mean {metric}: {best_errors_dict['baseline_model']}")

if best_model_error < best_errors_dict['baseline_model']:
    print(f"The best model '{best_model_name}' outperforms the baseline.")
else:
    print(f"The baseline model outperforms the best model '{best_model_name}'.")
print(datetime.datetime.now())
