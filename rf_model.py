
from sklearn.ensemble import RandomForestRegressor

def rf_model(X_train, y_train, X_test, params):

    if params is None:
        params = {
            'n_estimators': 50,
            'max_depth': 10,
        }

    model = RandomForestRegressor(**params, criterion='squared_error')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


param_groups = {
    'group1': { 
        'n_estimators': [25,50,100, 200],
        'max_depth': [None, 5, 10,20,40]
    },
    'group2': {  
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2,3],
        'max_features': ['sqrt', 'log2', None]  
}
}



model_type = 'multivariate'
