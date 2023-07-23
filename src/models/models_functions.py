import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_prep.pre_process import split_for_testing
from src.models.lasso import run_lasso
from src.models.xgboost_model import run_xgboost


def remove_outliers(X: pd.DataFrame, y: pd.DataFrame):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3-q1
    lower_fence = q1 - (1.5*iqr)
    higher_fence = q3 + (1.5*iqr)
    X = X[(y > lower_fence) & (y < higher_fence)]
    y = y[(y > lower_fence) & (y < higher_fence)]
    return X, y

def scale(X1, X2):
    numeric_features = X1.select_dtypes(include='float64', exclude='int64')
    scaler = StandardScaler()
    scaler.fit(X1.loc[:, numeric_features.columns])
    X1.loc[:, numeric_features.columns] = scaler.transform(X1.loc[:, numeric_features.columns])
    X2.loc[:, numeric_features.columns] = scaler.transform(X2.loc[:, numeric_features.columns])
    return X1, X2


# TODO - fix it - needs to be split into train and validation (not test) -> train_validation_split
def prepare_model_data(X: pd.DataFrame, y: pd.DataFrame, outliers=False):
    if outliers:
        X, y = remove_outliers(X, y)

    X_train, X_test, y_train, y_test = split_for_testing(X, y)

    X_train, X_test = scale(X_train, X_test)

    return X_train, X_test, y_train, y_test


def model(X: pd.DataFrame, y: pd.DataFrame, model_name: str, data_name: str, best_param=None, save_plots=False):
    print(f'Running {model_name} for {data_name}')

    X_train, X_test, y_train, y_test = prepare_model_data(X, y)

    if best_param is None:
        best_param = {}

    if model_name == 'lasso':
        r2, mse_score, spearman = run_lasso(X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)
    elif model_name == 'xgboost':
        r2, mse_score, spearman = run_xgboost(X_train, X_test, y_train, y_test, data_title=data_name,
                                              Best_param=best_param, save_plots=save_plots)
    else:
        raise Exception(f'No such model: {model_name}')
    return r2, mse_score, spearman

