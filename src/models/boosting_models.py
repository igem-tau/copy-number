import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMRegressor, plot_importance
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
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

def run_trees_model(model_name, X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if model_name == 'XGBoost':
        Best_param.pop('callbacks', None)
        Best_param['n_estimators'] = 1000
        if Best_param is not None:
            model = xgb.XGBRegressor(**Best_param)
        else:
            model = xgb.XGBRegressor()

    elif model_name == 'CatBoostRegressor':
        if Best_param is not None:
            model = CatBoostRegressor(**Best_param, allow_writing_files=False)
        else:
            model = CatBoostRegressor(allow_writing_files=False)

    elif model_name == 'RandomForest':
        if Best_param is not None:
            model = RandomForestRegressor(**Best_param)
        else:
            model = RandomForestRegressor()\

    elif model_name == 'LGBMRegressor':
        if Best_param is not None:
            model = LGBMRegressor(**Best_param)
        else:
            model = LGBMRegressor()

    else:
        raise ValueError(
            'models: models accepts only the following values: "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "Random Forest"')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 value for {model_name}: {r2}')
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f'MAE value for {model_name}: {mae_score}')
    pearson, _ = pearsonr(y_test, y_pred)
    print(f'pearson correlation value for {model_name}: {pearson}')
    spearman, _ = spearmanr(y_test, y_pred)
    print(f'spearman correlation value for {model_name}: {spearman}')

    if save_plots:
        if model_name == 'XGBoost':
            ax = xgb.plot_importance(model, max_num_features=20, title=f'{data_title} feature importance - XGBoost')
            ax.figure.savefig(Path(FIGURES_PATH, f'XGBoost feature importance {data_title}.jpg'))
        elif model_name =='CatBoostRegressor':
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
            plt.title('Feature Importance')
            plt.savefig(Path(FIGURES_PATH, f'CatBoost feature importance {data_title}.jpg'))
        elif model_name == 'RandomForest':
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
            plt.title('Feature Importance')
            plt.savefig(Path(FIGURES_PATH, f'Random Forest feature importance {data_title}.jpg'))
        elif model_name =='LGBMRegressor':
            ax = plot_importance(model, max_num_features=20,
                                 title=f'{data_title} feature importance - LGBMRegressor')
            ax.figure.savefig(Path(FIGURES_PATH, f'LGBMRegressor feature importance {data_title}.jpg'))

        # evaluation plot
        f, ax = plt.subplots()
        plt.scatter(y_test, y_pred)
        plt.axline((0, 0), slope=1)
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.text(0.8, 0.1, 'pearson correlation=%.4f' % pearson, transform=ax.transAxes)
        plt.text(0.8, 0.2, 'MAE=%.4f' % mae_score, transform=ax.transAxes)
        plt.title(f'{model_name} - {data_title}')
        if save_plots:
            plt.savefig(Path(FIGURES_PATH, f'{model_name} evaluation {data_title}.jpg'))

    return model, r2, mae_score, pearson, spearman, y_pred