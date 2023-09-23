# Fixed Import with load Diabitis instead of load boston
import re
from functools import partial
from tqdm import tqdm
import optuna
from BorutaShap import BorutaShap
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectFromModel
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from pathlib import Path
from sklearn.svm import SVR
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

from src.models.models_functions import scale
from src.utils import get_current_file_parent_path, get_current_date
from joblib import dump, load

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
model_names = ['NN', 'Ridge', 'Lasso', 'ElasticNet', 'XGBoost', 'CatBoostRegressor', 'LGBMRegressor']


def make_model(X_tr, X_va, y_tr, y_va, regressor_name: str, params):
    if regressor_name == 'Ridge':
        model = Ridge(**params)

    elif regressor_name == 'Lasso':
        model = Lasso(**params)

    elif regressor_name == 'ElasticNet':
        model = ElasticNet(**params)

    elif regressor_name == 'LGBMRegressor':

        X_tr = X_tr.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_va = X_va.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        model = LGBMRegressor(**params)

    elif regressor_name == 'RandomForest':
        model = RandomForestRegressor(**params)

    elif regressor_name == 'XGBoost':
        model = XGBRegressor(**params)

    elif regressor_name == 'CatBoostRegressor':
        model = CatBoostRegressor(**params)

    elif regressor_name == 'SVR':
        model = SVR(**params)

    elif regressor_name == 'NN':
        model = MLPRegressor(**params)

    model.fit(X_tr, y_tr)
    y_pred_train = model.predict(X_tr)
    y_pred_val = model.predict(X_va)
    r2_train = r2_score(y_tr, y_pred_train)
    r2_val = r2_score(y_va, y_pred_val)
    pearson_train, _ = pearsonr(y_tr, y_pred_train)
    pearson_val, _ = pearsonr(y_va, y_pred_val)
    mse_train = mean_squared_error(y_tr, y_pred_train)
    mse_val = mean_squared_error(y_va, y_pred_val)
    mae_train = mean_absolute_error(y_tr, y_pred_train)
    mae_val = mean_absolute_error(y_va, y_pred_val)
    return regressor_name, pearson_train, pearson_val, mae_train, mae_val, mse_train, mse_val, r2_train, r2_val


def get_hyper_parameters(trial=None, regressor_name=None):
    if regressor_name == 'Ridge':
        params = dict(alpha=trial.suggest_float("alpha", 0, 5),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      tol=trial.suggest_float("tol", 1e-6, 0.001, log=True),
                      solver=trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]))
        regressor_obj = Ridge(**params)

    elif regressor_name == 'Lasso':
        params = dict(alpha=trial.suggest_float("alpha", 0, 1),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      tol=trial.suggest_float("tol", 1e-4, 0.01, log=True),
                      selection=trial.suggest_categorical("selection", ["cyclic", "random"]),
                      warm_start=trial.suggest_categorical('warm_start', [True, False]))
        regressor_obj = Lasso(**params)

    elif regressor_name == 'ElasticNet':
        params = dict(alpha=trial.suggest_float("alpha", 0, 1),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      l1_ratio=trial.suggest_float('l1_ratio', 0, 0.5),
                      tol=trial.suggest_float("tol", 1e-5, 0.001, log=True),
                      selection=trial.suggest_categorical("selection", ["cyclic", "random"]),
                      warm_start=trial.suggest_categorical('warm_start', [True, False]))
        regressor_obj = ElasticNet(**params)

    # elif regressor_name == 'GradientBoosting':
    #     params = dict(
    #         loss = trial.suggest_categorical('loss', ["squared_error", "absolute_error", "huber", "quantile"]),
    #         learning_rate= trial.suggest_float('learning_rate', 0.005, 0.5),
    #         subsample= trial.suggest_float('subsample', 0.5, 1.0),
    #         criterion = trial.suggest_categorical('criterion', ["friedman_mse", "squared_error"]),
    #         n_estimators = trial.suggest_int('n_estimators', 50, 200),
    #         min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
    #         min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20),
    #         max_depth = trial.suggest_int('max_depth', 2, 20),
    #         alpha = trial.suggest_float("alpha", 0.5, 1),
    #         tol = trial.suggest_float("tol", 1e-4, 0.01, log=True),
    #         max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
    #         warm_start=trial.suggest_categorical('warm_start', [True, False]))
    #     regressor_obj = GradientBoostingRegressor(**params)

    elif regressor_name == 'LGBMRegressor':
        params = dict(boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                      num_leaves=trial.suggest_int('num_leaves', 15, 50),
                      max_depth=trial.suggest_categorical('max_depth', [-1, 5, 10, 20]),
                      learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                      reg_alpha=trial.suggest_float('reg_alpha', 0.01, 1.0),
                      reg_lambda=trial.suggest_float('reg_lambda', 0.01, 1.0),
                      min_split_gain=trial.suggest_float('min_split_gain', 0, 0.5),
                      min_child_samples=trial.suggest_int('min_child_samples', 10, 30),
                      subsample=trial.suggest_float('subsample', 0.5, 1),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.01, 1.0))
        regressor_obj = LGBMRegressor(**params)

    elif regressor_name == 'RandomForest':
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 500),
            max_depth=trial.suggest_int("max_depth", 10, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            criterion=trial.suggest_categorical('criterion', ["friedman_mse"]),
            max_features=trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
            warm_start=trial.suggest_categorical('warm_start', [True, False]))
        regressor_obj = RandomForestRegressor(**params)

    elif regressor_name == 'XGBoost':
        params = dict(
            max_depth=trial.suggest_int('max_depth', 1, 15),
            learning_rate=trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
            n_estimators=trial.suggest_int('n_estimators', 50, 1000),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            gamma=trial.suggest_float('gamma', 0.3, 1.0),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.01, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.01, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.01, 1.0)
        )
        regressor_obj = XGBRegressor(**params)

    elif regressor_name == 'CatBoostRegressor':
        params = dict(
            silent=trial.suggest_categorical('silent', [True]),
            loss_function=trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
            depth=trial.suggest_int('depth', 5, 16),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.01, 5.0),
            subsample=trial.suggest_float("subsample", 0.05, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.05, 0.8),
            min_child_samples=trial.suggest_categorical('min_child_samples', [1, 4, 8, 16]),
            grow_policy=trial.suggest_categorical('grow_policy', ['Depthwise', 'SymmetricTree', 'Lossguide']),
        )
        regressor_obj = CatBoostRegressor(**params)

    elif regressor_name == 'SVR':
        params = dict(
            kernel=trial.suggest_categorical('kernel', ["linear", "poly", "rbf", "sigmoid"]),
            degree=trial.suggest_int('degree', 2, 5),
            gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            tol=trial.suggest_float("tol", 1e-3, 0.1, log=True),
            C=trial.suggest_float('C', 1, 3),
            epsilon=trial.suggest_float('epsilon', 0.001, 1, log=True)
        )
        regressor_obj = SVR(**params)

    elif regressor_name == 'NN':
        params = dict(
            hidden_layer_sizes=trial.suggest_int('hidden_layer_sizes', 100, 400),
            solver=trial.suggest_categorical('solver', ["lbfgs", "sgd", "adam"]),
            activation=trial.suggest_categorical('activation', ["logistic", "tanh", "relu"]),
            alpha=trial.suggest_float("alpha", 1e-5, 0.001, log=True),
            learning_rate=trial.suggest_categorical('learning_rate', ["constant", "invscaling", "adaptive"]),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-6, 0.001, log=True),
            tol=trial.suggest_float("tol", 1e-5, 0.001, log=True),
            momentum=trial.suggest_float("momentum", 0.1, 0.9),
            beta_1=trial.suggest_float("beta_1", 0.5, 0.999),
            beta_2=trial.suggest_float("beta_2", 0.3, 0.9),
            epsilon=trial.suggest_float("epsilon", 1e-8, 1e-4, log=True)
        )
        regressor_obj = MLPRegressor(**params)
    return regressor_obj, params


def objective(trial, X_train, y_train, X_val, y_val, regressor):
    regressor_obj, params = get_hyper_parameters(trial, regressor)
    if regressor == 'LGBMRegressor':
        X_train = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_val = X_val.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)
    return r2_score(y_val, y_pred)


def model_selection(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                    rna_type: str) -> Dict:
    selected_model_params_path = Path(DATA_PATH, f'RNA{rna_type}_Selected_Models_params.joblib')
    model_params_path = Path(DATA_PATH, f'RNA{rna_type}_df_Models_params.joblib')
    if selected_model_params_path.exists():
        params_dict = load(selected_model_params_path)
    else:
        df_models = pd.DataFrame(data=None,
                                 columns=['Algorithm', 'pearson_train', 'pearson_val', 'mae_train', 'mae_val'])
        X_train_scaled, X_val_scaled = scale(X_train, X_val)
        params_dict = {}
        for model_name in tqdm(model_names):
            print(f"Running: {model_name} for model selection")
            trails = {'Ridge': 150, 'Lasso': 150, 'ElasticNet': 150, 'GradientBoosting': 100, 'LGBMRegressor': 200,
                      'XGBoost': 200,
                      'CatBoostRegressor': 100, 'SVR': 50, 'NN': 100}
            trails = {'Ridge': 200, 'Lasso': 2, 'ElasticNet': 2, 'GradientBoosting': 2, 'LGBMRegressor': 2,
                      'XGBoost': 1,
                      'CatBoostRegressor': 1, 'SVR': 2, 'NN': 1}
            study = optuna.create_study(direction='maximize')
            if Path(DATA_PATH, f'{model_name}_study').exists():
                last_study = load(Path(DATA_PATH, f'{get_current_date()}_{model_name}_study'))
                study.add_trials(last_study.trials)
            study.optimize(partial(objective, X_train=X_train_scaled, y_train=y_train, X_val=X_val_scaled, y_val=y_val,
                                   regressor=model_name),
                           n_trials=trails[model_name])

            params = study.best_trial.params
            params_dict[model_name] = params
            dump(study, Path(DATA_PATH, f'{get_current_date()}_{model_name}_study'))

            model_name, pearson_train, pearson_val, mae_train, mae_val, _, _, _, _ = make_model(X_train, X_val, y_train,
                                                                                                y_val, model_name,
                                                                                                params)
            df_models.loc[len(df_models.index)] = [model_name, pearson_train, pearson_val, mae_train, mae_val]
            print(f"Finished: {model_name} for model selection")

        dump(params_dict, selected_model_params_path)
        dump(df_models, model_params_path)

        print('Creating plot for model selection')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(name='pearson train', x=df_models.Algorithm, y=df_models.pearson_train),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(name='pearson validation', x=df_models.Algorithm, y=df_models.pearson_val),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(name='MAE train', x=df_models.Algorithm, y=df_models.mae_train),
            secondary_y=True,
        )
        fig.add_trace(
            go.Bar(name='MAE validation', x=df_models.Algorithm, y=df_models.mae_val),
            secondary_y=True,
        )

        fig.update_layout(template='plotly_white', title='Pearson correlation and Mean Absolute Error (MAE) for train '
                                                         'and validation sets', title_x=0.5, yaxis=dict(
            title=dict(text="Pearson correlation"),
            side="left"), yaxis2=dict(
            title=dict(text="MAE"),
            side="right"))

        with open(Path(DATA_PATH, f'{get_current_date()}_RNA{rna_type}_model_selection_graphs.html'), 'w') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return params_dict


'''
https://github.com/Ekeany/Boruta-Shap
https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
'''


def feature_selection(RNA_X, RNA_y, param_dict, models, rna_type):
    """
    Feature Selection for RNAp or RNAi.

    Accept: DataFrame of training data and dataframe for test.
    Return: (Subset of data_train model with accepted features only, Array of accepted features, Array of Denied Features).

    Using Data vendding based on correlation between features and dropping uncorrelated ones that are under the minimum.
    Using BorutaShap as model for feature selection (Wrapper Method).
    """
    # TODO - rename to a clearer name, since it is not the file used at the end, to read the selected features
    filename = f'RNA{rna_type}_Selected_Features.joblib'

    if Path(DATA_PATH, filename).exists():
        models_data = load(Path(DATA_PATH, filename))
    else:
        print('Running: features selection - features and copy number')
        # feature vetting: select features based on correlations only
        # correlation between features and copy number (maximal) with MI
        mi = mutual_info_regression(RNA_X, RNA_y, discrete_features=(RNA_X.dtypes == 'int64'))
        RNA_X_new = RNA_X.iloc[:, (mi > (mi.mean()))]
        new_mi = mi[(mi > (mi.mean()))]

        # correlation between feature-feature (minimal) with MI
        print('Running: features selection - feature-feature')
        new_features = pd.Series(RNA_X_new.columns)
        corr_matrix = np.zeros((len(new_features), len(new_features)))
        for i in range(len(new_features)):
            for j in range(len(new_features)):
                if i >= j:
                    continue
                discrete_features_bool = True if RNA_X_new[new_features[i]].dtype == 'int64' else False
                if RNA_X_new[new_features[j]].dtype == 'int64':  # y is categorical
                    corr_matrix[i, j] = mutual_info_classif(RNA_X_new[new_features[i]].to_numpy().reshape(-1, 1),
                                                            RNA_X_new[new_features[j]],
                                                            discrete_features=discrete_features_bool)
                else:  # y is numerical
                    corr_matrix[i, j] = mutual_info_regression(RNA_X_new[new_features[i]].to_numpy().reshape(-1, 1),
                                                               RNA_X_new[new_features[j]],
                                                               discrete_features=discrete_features_bool)

        il1 = np.tril_indices(len(new_features))
        corr_matrix[il1] = np.nan

        (row, col) = (corr_matrix > np.nanquantile(corr_matrix, 0.99)).nonzero()  # TODO: think of different condition

        while len(row) > 0:
            values = np.array([row[0], col[0]])  # first pair
            inx = new_mi[values].argmin()  # find the feature with less correlation to the copy number
            new_features.drop(values[inx], inplace=True)  # erase from features Series

            # erase from row and col
            cur_inx_row = (row == values[inx]).nonzero()
            cur_inx_col = (col == values[inx]).nonzero()
            row = np.delete(row, np.concatenate((cur_inx_row, cur_inx_col), axis=1))
            col = np.delete(col, np.concatenate((cur_inx_row, cur_inx_col), axis=1))

        RNA_X_new = RNA_X_new.loc[:, new_features]

        # feature selection - Boruta Sharp.
        models_data = {}
        for model in models:
            if model == 'XGBoost':
                print('Running: XGBoost for feature selection using boruta shap')
                estimator = XGBRegressor(**param_dict[model])
            elif model == 'CatBoostRegressor':
                estimator = CatBoostRegressor(**param_dict[model])
                print('Running: CatBoost for feature selection using boruta shap')
            else:
                raise ValueError(
                    'feature_selection: models accepts only the following values: "XGBoost" or "CatBoostRegressor"')

            Feature_Selector = BorutaShap(model=estimator,
                                          importance_measure='shap',
                                          classification=False)

            Feature_Selector.fit(X=RNA_X_new, y=RNA_y, n_trials=200, sample=False,  # TODO: sample_fraction=0.85,?
                                 train_or_test='test', normalize=False,
                                 verbose=True)

            # Return Values :
            features_to_remove = Feature_Selector.features_to_remove
            features_to_accept = Feature_Selector.accepted
            subset_of_data = Feature_Selector.Subset()

            data = {f'RNA{rna_type}_train_FS': subset_of_data, 'selected_features': features_to_accept,
                    'removed_features': features_to_remove}
            models_data[model] = data

        dump(models_data, Path(DATA_PATH, filename))

    return models_data
