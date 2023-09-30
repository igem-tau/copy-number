import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils import get_current_file_parent_path
from typing import Optional
import warnings
import xgboost as xgb
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import pearsonr


warnings.simplefilter(action='ignore', category=FutureWarning)

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
FIGURES_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'figures')


def run_xgboost(X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if Best_param is not None:
        # Best_param['max_depth'], Best_param['n_estimators'] = int(Best_param['max_depth']), int(Best_param['n_estimators'])
        Best_param.pop('callbacks', None)
        Best_param['n_estimators'] = 1000
        xgb_model = xgb.XGBRegressor(**Best_param)
    else:
        xgb_model = xgb.XGBRegressor()

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 value for xgboost: {r2}')
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f'MAE value for xgboost: {mae_score}')
    pearson, _ = pearsonr(y_test, y_pred)
    print(f'pearson correlation value for xgboost: {pearson}')
    spearman, _ = spearmanr(y_test, y_pred)
    print(f'spearman correlation value for xgboost: {spearman}')


    if save_plots:
        ax = xgb.plot_importance(xgb_model, max_num_features=20, title=f'{data_title} feature importance - XGBoost')
        ax.figure.savefig(Path(FIGURES_PATH, f'XGBoost feature importance {data_title}.jpg'))

    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_test, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'pearson correlation=%.4f' % pearson, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MAE=%.4f' % mae_score, transform=ax.transAxes)
    plt.title(f'XGBoost - {data_title}')
    if save_plots:
        plt.savefig(Path(FIGURES_PATH, f'XGBoost evaluation {data_title}.jpg'))

    return xgb_model, r2, mae_score, pearson, spearman, y_pred

def run_catboost(X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if Best_param is not None:
        # Best_param['max_depth'], Best_param['n_estimators'] = int(Best_param['max_depth']), int(Best_param['n_estimators'])
        catb_model = CatBoostRegressor(**Best_param)
    else:
        catb_model = CatBoostRegressor()

    catb_model.fit(X_train, y_train)
    y_pred = catb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 value for catboost: {r2}')
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f'MAE value for catboost: {mae_score}')
    pearson, _ = pearsonr(y_test, y_pred)
    print(f'pearson correlation value for catboost: {pearson}')
    spearman, _ = spearmanr(y_test, y_pred)
    print(f'spearman correlation value for catboost: {spearman}')

    feature_importance = catb_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')

    if save_plots:
        plt.savefig(Path(FIGURES_PATH, f'CatBoost feature importance {data_title}.jpg'))

    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_test, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'R2=%.4f' % r2, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MAE=%.4f' % mae_score, transform=ax.transAxes)
    plt.title(f'CatBoost - {data_title}')
    if save_plots:
        plt.savefig(Path(FIGURES_PATH, f'CatBoost evaluation {data_title}.jpg'))

    return catb_model, r2, mae_score, pearson, spearman, y_pred

def run_rf(X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if Best_param is not None:
        # Best_param['max_depth'], Best_param['n_estimators'] = int(Best_param['max_depth']), int(Best_param['n_estimators'])
        rf_model = RandomForestRegressor(**Best_param)
    else:
        rf_model = RandomForestRegressor()

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 value for Random Forest: {r2}')
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f'MAE value for Random Forest: {mae_score}')
    pearson, _ = pearsonr(y_test, y_pred)
    print(f'pearson correlation value for Random Forest: {pearson}')
    spearman, _ = spearmanr(y_test, y_pred)
    print(f'spearman correlation value for Random Forest: {spearman}')

    feature_importance = rf_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')

    if save_plots:
        plt.savefig(Path(FIGURES_PATH, f'Random Forest feature importance {data_title}.jpg'))

    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_test, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'R2=%.4f' % r2, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MAE=%.4f' % mae_score, transform=ax.transAxes)
    plt.title(f'Random Forest - {data_title}')
    if save_plots:
        plt.savefig(Path(FIGURES_PATH, f'Random Forest evaluation {data_title}.jpg'))

    return rf_model, r2, mae_score, pearson, spearman, y_pred
