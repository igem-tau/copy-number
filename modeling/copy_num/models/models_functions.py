from modeling.copy_num.models.lasso import run_lasso
from modeling.copy_num.models.xgboost_model import run_xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def is_high_copy_number(copy_number: 'pd.Series[int]') -> 'pd.Series[int]':
    percentage = 0.2
    n = int(copy_number.shape[0] * percentage)
    high_cp = copy_number.nlargest(n)
    return (copy_number >= high_cp.min()).astype(int)


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

def prepare_model_data(X: pd.DataFrame, y: pd.DataFrame, outliers=False):
    if outliers:
        X, y = remove_outliers(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0,
                                                        stratify=is_high_copy_number(y))

    X_train, X_test = scale(X_train, X_test)

    return X_train, X_test, y_train, y_test


def model(X: pd.DataFrame, y: pd.DataFrame, model_name: str, data_name: str, best_param=None, save_plots=False):
    print(f"Running {model_name} for {data_name}")

    X_train, X_test, y_train, y_test = prepare_model_data(X, y)

    if best_param is None:
        best_param = {}

    if model_name == "lasso":
        r2, mse_score, spearman = run_lasso(X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)
    elif model_name == "xgboost":
        r2, mse_score, spearman = run_xgboost(X_train, X_test, y_train, y_test, data_title=data_name,
                                              Best_param=best_param, save_plots=save_plots)
    else:
        raise Exception(f"No such model: {model_name}")
    return r2, mse_score, spearman


def train_validation_test_split(X, y, random_stat):
    stratify_col = pd.DataFrame(list(is_high_copy_number(y)), columns=['stratify'])
    X = pd.concat([X.reset_index(), stratify_col], axis=1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_stat, stratify=X['stratify'])
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_stat,
                                                        stratify=X_temp['stratify'])

    return (X_train.drop('stratify', axis=1).drop('index', axis=1), X_valid.drop('stratify', axis=1).drop('index', axis=1),
            X_test.drop('stratify', axis=1).drop('index', axis=1), y_train, y_valid, y_test)
