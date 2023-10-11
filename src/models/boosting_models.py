from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import get_current_file_parent_path, estimate_pred
from typing import Optional
import warnings
import xgboost as xgb
from catboost import CatBoostRegressor
from src.consts import RANDOM_STATE

warnings.simplefilter(action='ignore', category=FutureWarning)

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
FIGURES_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'figures')


def run_model(model_name, X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
              save_plots: bool = False):
    if model_name == 'XGBoost':
        Best_param.pop('callbacks', None)
        Best_param['n_estimators'] = 1000
        if Best_param is not None:
            model = xgb.XGBRegressor(**Best_param, random_state=RANDOM_STATE)
        else:
            model = xgb.XGBRegressor(random_state=RANDOM_STATE)

    elif model_name == 'CatBoostRegressor':
        if Best_param is not None:
            model = CatBoostRegressor(**Best_param, allow_writing_files=False, random_state=RANDOM_STATE)
        else:
            model = CatBoostRegressor(allow_writing_files=False, random_state=RANDOM_STATE)

    elif model_name == 'RandomForest':
        if Best_param is not None:
            model = RandomForestRegressor(**Best_param, random_state=RANDOM_STATE)
        else:
            model = RandomForestRegressor(random_state=RANDOM_STATE)

    elif model_name == 'LGBMRegressor':
        if Best_param is not None:
            model = LGBMRegressor(**Best_param, random_state=RANDOM_STATE)
        else:
            model = LGBMRegressor(random_state=RANDOM_STATE)

    else:
        raise ValueError(
            'models: models accepts only the following values: "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "Random Forest"')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2, mae_score, pearson, spearman = estimate_pred(y_test, y_pred, model_name, data_title, model, save_plots)
    return model, r2, mae_score, pearson, spearman, y_pred
