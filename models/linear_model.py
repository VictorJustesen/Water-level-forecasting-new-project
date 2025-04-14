
from sklearn.linear_model import LinearRegression

def linear_model(X_train, y_train, X_test, params=None):
   
    if params is None:
        params = {'fit_intercept': True}

    model = LinearRegression(**params,)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred

param_groups = {
    'group1': {
        'fit_intercept': [True, False]
    }
}

model_type = 'multivariate'
