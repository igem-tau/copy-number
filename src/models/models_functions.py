import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.lasso import run_lasso
from src.models.boosting_models import run_model


def remove_outliers(X: pd.DataFrame, y: pd.DataFrame):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3-q1
    lower_fence = q1 - (1.5*iqr)
    higher_fence = q3 + (1.5*iqr)
    X = X[(y > lower_fence) & (y < higher_fence)]
    y = y[(y > lower_fence) & (y < higher_fence)]
    return X, y


def scale(X1, X2):
    scaler = StandardScaler()
    scaler.fit(X1)
    X1 = pd.DataFrame(scaler.transform(X1), columns=X1.columns)
    X2 = pd.DataFrame(scaler.transform(X2), columns=X2.columns)
    return X1, X2


# TODO - should NOT be used
def prepare_model_data(X: pd.DataFrame, y: pd.DataFrame, outliers=False):
    if outliers:
        X, y = remove_outliers(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    X_train, X_test = scale(X_train, X_test)

    return X_train, X_test, y_train, y_test


def model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, model_name: str, data_name: str, best_param=None, save_plots=False):
    print(f'Running {model_name} for {data_name} with {len(X_train.columns)} features')
    X_train, X_test = scale(X_train, X_test)
    if best_param is None:
        best_param = {}

    if model_name == 'lasso':
        model, r2, mae_score, spearman = run_lasso(X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)
    else:
        model, r2, mae_score, pearson, spearman, y_pred = run_model(model_name, X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)

    return model, r2, mae_score, pearson, spearman, y_pred

