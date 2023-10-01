import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.lasso import run_lasso
from src.models.boosting_models import run_trees_model
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import get_current_file_parent_path


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
FIGURES_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'figures')


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

def estimate_pred(y_true, y_pred, name_of_model):
    r2 = r2_score(y_true, y_pred)
    print(f'R^2 value for {name_of_model}: {r2}')
    mae_score = mean_absolute_error(y_true, y_pred)
    print(f'MAE value for {name_of_model}: {mae_score}')
    pearson, _ = pearsonr(y_true, y_pred)
    print(f'pearson correlation value for {name_of_model}: {pearson}')
    spearman, _ = spearmanr(y_true, y_pred)
    print(f'spearman correlation value for {name_of_model}: {spearman}')

    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_true, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'pearson correlation=%.4f' % pearson, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MAE=%.4f' % mae_score, transform=ax.transAxes)
    plt.title(f'{name_of_model} evaluation')
    plt.savefig(Path(FIGURES_PATH, f'{name_of_model} evaluation.jpg'))


def model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, model_name: str, data_name: str, best_param=None, save_plots=False):
    print(f'Running {model_name} for {data_name} with {len(X_train.columns)} features')
    X_train, X_test = scale(X_train, X_test)
    if best_param is None:
        best_param = {}

    if model_name == 'lasso':
        model, r2, mse_score, spearman = run_lasso(X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)
    else:
        model, r2, mae_score, pearson, spearman, y_pred = run_trees_model(model_name, X_train, X_test, y_train, y_test, data_title=data_name,
                                            Best_param=best_param, save_plots=save_plots)

    return model, r2, mae_score, pearson, spearman, y_pred

