import re
from src.consts import RANDOM_STATE
from functools import partial
from tqdm import tqdm
import optuna
from eBoruta import eBoruta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict

from src.models.models_functions import scale
from src.utils import get_current_file_parent_path, get_current_date
from joblib import dump, load, Parallel, delayed

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
model_names = ['NN', 'Ridge', 'Lasso', 'ElasticNet', 'XGBoost', 'CatBoostRegressor', 'LGBMRegressor', 'RandomForest']


def make_model(X_tr, X_va, y_tr, y_va, regressor_name: str, params):
    if regressor_name == 'Ridge':
        model = Ridge(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'Lasso':
        model = Lasso(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'ElasticNet':
        model = ElasticNet(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'LGBMRegressor':
        X_tr = X_tr.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_va = X_va.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        model = LGBMRegressor(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'RandomForest':
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'XGBoost':
        model = XGBRegressor(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'CatBoostRegressor':
        model = CatBoostRegressor(**params, allow_writing_files=False, random_state=RANDOM_STATE)

    elif regressor_name == 'NN':
        model = MLPRegressor(**params, random_state=RANDOM_STATE)

    model.fit(X_tr, y_tr)
    y_pred_train = model.predict(X_tr)
    y_pred_val = model.predict(X_va)
    r2_train = r2_score(y_tr, y_pred_train)
    r2_val = r2_score(y_va, y_pred_val)
    pearson_train, _ = pearsonr(y_tr, y_pred_train)
    pearson_val, _ = pearsonr(y_va, y_pred_val)
    spearman_train, _ = spearmanr(y_tr, y_pred_train)
    spearman_val, _ = spearmanr(y_va, y_pred_val)
    mse_train = mean_squared_error(y_tr, y_pred_train)
    mse_val = mean_squared_error(y_va, y_pred_val)
    return regressor_name, pearson_train, pearson_val, spearman_train, spearman_val, mse_train, mse_val, r2_train, r2_val


def get_hyper_parameters(trial, regressor_name):
    if regressor_name == 'Ridge':
        params = dict(alpha=trial.suggest_float("alpha", 0, 20),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      tol=trial.suggest_float("tol", 1e-6, 0.001, log=True),
                      solver=trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]))
        regressor_obj = Ridge(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'Lasso':
        params = dict(alpha=trial.suggest_float("alpha", 0, 5),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      tol=trial.suggest_float("tol", 1e-4, 0.01, log=True),
                      selection=trial.suggest_categorical("selection", ["cyclic", "random"]),
                      warm_start=trial.suggest_categorical('warm_start', [True, False]))
        regressor_obj = Lasso(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'ElasticNet':
        params = dict(alpha=trial.suggest_float("alpha", 0, 5),
                      fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                      l1_ratio=trial.suggest_float('l1_ratio', 0, 0.5),
                      tol=trial.suggest_float("tol", 1e-5, 0.001, log=True),
                      selection=trial.suggest_categorical("selection", ["cyclic", "random"]),
                      warm_start=trial.suggest_categorical('warm_start', [True, False]))
        regressor_obj = ElasticNet(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'LGBMRegressor':
        params = dict(verbose=-1,
                      boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                      num_leaves=trial.suggest_int('num_leaves', 15, 30),
                      max_depth=trial.suggest_categorical('max_depth', [-1, 5, 10, 20]),
                      learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                      reg_alpha=trial.suggest_float('reg_alpha', 0.01, 1.0),
                      reg_lambda=trial.suggest_float('reg_lambda', 0.01, 1.0),
                      min_split_gain=trial.suggest_float('min_split_gain', 0, 0.5),
                      min_child_samples=trial.suggest_int('min_child_samples', 10, 30),
                      subsample=trial.suggest_float('subsample', 0.5, 1),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.01, 1.0))
        regressor_obj = LGBMRegressor(**params, random_state=RANDOM_STATE)

    elif regressor_name == 'RandomForest':
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 500),
            max_depth=trial.suggest_int("max_depth", 10, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            criterion=trial.suggest_categorical('criterion', ["friedman_mse"]),
            max_features=trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
            # warm_start=trial.suggest_categorical('warm_start', [True, False])
        )
        regressor_obj = RandomForestRegressor(**params, random_state=RANDOM_STATE)

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
        regressor_obj = XGBRegressor(**params, random_state=RANDOM_STATE)

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
        regressor_obj = CatBoostRegressor(**params, allow_writing_files=False, random_state=RANDOM_STATE)

    elif regressor_name == 'NN':
        params = dict(max_iter=10000,
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
        regressor_obj = MLPRegressor(**params, random_state=RANDOM_STATE)
    else:
        raise ValueError(
            'hyperparametes: models accepts only the following values: "NN", "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "RandomForest"')

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
                                 columns=['Algorithm', 'pearson_train', 'pearson_val', 'spearman_train', 'spearman_val'])
        X_train_scaled, X_val_scaled = scale(X_train, X_val)
        params_dict = {}
        max_trials_per_optimization_cycle = 5
        for model_name in tqdm(model_names):
            study_file_name = f'RNA{rna_type}_{model_name}_study'
            print(f"Running: {model_name} for model selection")
            trials = {'Ridge': 150, 'Lasso': 150, 'ElasticNet': 150, 'LGBMRegressor': 200,
                      'XGBoost': 200,
                      'CatBoostRegressor': 100, 'NN': 100, 'RandomForest': 200}

            study = optuna.create_study(direction='maximize')
            if Path(DATA_PATH, study_file_name).exists():
                last_study = load(Path(DATA_PATH, study_file_name))
                study.add_trials(last_study.trials)

            num_trials_left = max(0, trials[model_name] - len(study.trials))
            while num_trials_left > 0:
                current_num_trials = min(num_trials_left, max_trials_per_optimization_cycle)
                study.optimize(
                    partial(objective, X_train=X_train_scaled, y_train=y_train, X_val=X_val_scaled, y_val=y_val,
                            regressor=model_name),
                    n_trials=current_num_trials)
                dump(study, Path(DATA_PATH, study_file_name), compress=True)
                num_trials_left -= current_num_trials

            params = study.best_trial.params
            params_dict[model_name] = params
            dump(params_dict, selected_model_params_path, compress=True)

            if model_params_path.exists():
                df_models = load(model_params_path)
            if model_name not in df_models['Algorithm'].values.tolist():
                model_name, pearson_train, pearson_val, spearman_train, spearman_val, _, _, _, _ = make_model(X_train, X_val, y_train,
                                                                                                    y_val, model_name,
                                                                                                    params)
                df_models.loc[len(df_models.index)] = [model_name, pearson_train, pearson_val, spearman_train, spearman_val]
                dump(df_models, model_params_path, compress=True)
            print(f"Finished: {model_name} for model selection")

        print('Creating plot for model selection')
        fig = go.Figure(
            data=[
                go.Bar(name='pearson train', x=df_models.Algorithm, y=df_models.pearson_train, yaxis='y',
                       offsetgroup=1),
                go.Bar(name='pearson validation', x=df_models.Algorithm, y=df_models.pearson_val, yaxis='y',
                       offsetgroup=2),
                go.Bar(name='Spearman train', x=df_models.Algorithm, y=df_models.spearman_train, yaxis='y', offsetgroup=3,
                       visible='legendonly'),
                go.Bar(name='Spearman validation', x=df_models.Algorithm, y=df_models.spearman_val, yaxis='y', offsetgroup=4,
                       visible='legendonly')
            ],
            layout=dict(
                template='plotly_white',
                title='Pearson and Spearman correlations for train and validation sets',
                title_x=0.5,
                yaxis=dict(title='Pearson / Spearman correlation'),
                # yaxis2=dict(title='Spearman correlationE', overlaying='y', side='left'),
                barmode='group'
            )
        )

        with open(Path(DATA_PATH, f'{get_current_date()}_RNA{rna_type}_model_selection_graphs.html'), 'w') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return params_dict


'''
https://github.com/Ekeany/Boruta-Shap
https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
'''


def feature_selection(RNA_X, RNA_y, param_dict, model, rna_type):
    """
    Feature Selection for RNAp or RNAi.

    Accept: DataFrame of training data and dataframe for test.
    Return: (Subset of data_train model with accepted features only, Array of accepted features, Array of Denied Features).

    Using Data vendding based on correlation between features and dropping uncorrelated ones that are under the minimum.
    Using BorutaShap as model for feature selection (Wrapper Method).
    """
    # TODO - rename to a clearer name, since it is not the file used at the end, to read the selected features
    filename = f'RNA{rna_type}_{model}_Selected_Features.joblib'

    if Path(DATA_PATH, filename).exists():
        features_to_accept = load(Path(DATA_PATH, filename))
    else:
        print('Running: features selection - features and copy number')
        # feature vetting: select features based on correlations only
        # correlation between features and copy number (maximal) with MI
        mi = mutual_info_regression(RNA_X, RNA_y, discrete_features=(RNA_X.dtypes == 'int64'), random_state=0)
        RNA_X_new = RNA_X.iloc[:, (mi > (mi.mean()))]
        new_mi = mi[(mi > (mi.mean()))]

        # parallel approach for feature-feature correlation
        # Define a function to calculate mutual information
        def calculate_mutual_information(feature1, feature2, discrete_features_bool):
            if feature2.dtype == 'int64':  # y is categorical
                paired_mi = mutual_info_classif(feature1.to_numpy().reshape(-1, 1), feature2,
                                                discrete_features=discrete_features_bool, random_state=0)
            else:  # y is numerical
                paired_mi = mutual_info_regression(feature1.to_numpy().reshape(-1, 1), feature2,
                                                   discrete_features=discrete_features_bool, random_state=0)
            return paired_mi

        def parallel_calculate_mi(i, j):
            feature1 = RNA_X_new[new_features[i]]
            feature2 = RNA_X_new[new_features[j]]
            discrete_features_bool = feature1.dtype == 'int64'
            paired_mi = calculate_mutual_information(feature1, feature2, discrete_features_bool)
            return float(paired_mi)

        # correlation between feature-feature (minimal) with MI
        print('Running: features selection - feature-feature')
        if Path(DATA_PATH, f'RNA{rna_type}_mutual_info_matrix.joblib').exists():
            corr_matrix = load(Path(DATA_PATH, f'RNA{rna_type}_mutual_info_matrix.joblib'))
            new_features = pd.Series(RNA_X_new.columns)
        else:
            new_features = pd.Series(RNA_X_new.columns)
            num_new_features = len(new_features)
            corr_matrix = np.zeros((num_new_features, num_new_features))

            # Create pairs of feature indices for upper triangle calculation
            upper_triangle_indices = np.triu_indices(num_new_features, k=1)
            pairs = zip(*upper_triangle_indices)

            # Parallelize the mutual information calculation for upper triangle
            mi_values_upper = Parallel(n_jobs=-2)(delayed(parallel_calculate_mi)(i, j) for i, j in
                                                  tqdm(pairs, total=((num_new_features ** 2 - num_new_features) // 2)))

            # Fill the correlation matrix for the upper triangle
            corr_matrix[upper_triangle_indices] = mi_values_upper

            # Transform the lower triangle to np.nan
            il1 = np.tril_indices(num_new_features)
            corr_matrix[il1] = np.nan

            dump(corr_matrix, Path(DATA_PATH, f'RNA{rna_type}_mutual_info_matrix.joblib'), compress=True)

        row, col = (corr_matrix > np.nanquantile(corr_matrix, 0.99)).nonzero()  # TODO: think of different condition

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

        # feature selection - Boruta Shap.

        if model == 'XGBoost':
            print('Running: XGBoost for feature selection using eboruta')
            estimator = XGBRegressor(**param_dict[model], random_state=RANDOM_STATE)
        elif model == 'CatBoostRegressor':
            estimator = CatBoostRegressor(**param_dict[model], allow_writing_files=False, random_state=RANDOM_STATE)
            print('Running: CatBoost for feature selection using eboruta')
        elif model == 'RandomForest':
            estimator = RandomForestRegressor(**param_dict[model], random_state=RANDOM_STATE)
            print('Running: Random Forest for feature selection using eboruta')
        elif model == 'LGBMRegressor':
            estimator = LGBMRegressor(**param_dict[model], random_state=RANDOM_STATE)
        else:
            raise ValueError(
                'feature_selection: models accepts only the following values: "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "RandomForest"')

        importance_getter = get_features_importance if model in ['CatBoostRegressor', 'LGBMRegressor'] else None
        eboruta = eBoruta(n_iter=300, classification=False, shap_check_additivity=False, shap_approximate=True,
                          importance_getter=importance_getter, verbose=1).fit(RNA_X_new, RNA_y, model=estimator)

        # Return Values :
        features = eboruta.features_
        features_to_accept = features.accepted

        if len(features_to_accept) == 0:
            try:
                features_importance = pd.DataFrame(
                    {'feature': estimator.feature_name_ if isinstance(estimator,
                                                                      LGBMRegressor) else estimator.feature_names_,
                     'importance': estimator.feature_importances_})
                features_importance = features_importance.sort_values(by='importance', ascending=False)
                features_to_accept = features_importance.head(10)['feature']
            except AttributeError:
                features_to_accept = ['pssm_score', 'dG_total', 'rpoD16_score', 'C__T_count', 'GTA_GC_count']

        if model == 'LGBMRegressor':
            original_columns = pd.Series(RNA_X.columns)
            lgbm_columns = original_columns.apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            features_to_accept = pd.DataFrame(zip(original_columns, lgbm_columns), columns=['before', 'after']).query(
                f'after.isin({list(features_to_accept)})')['before']
        features_to_accept = list(features_to_accept)

        dump(features_to_accept, Path(DATA_PATH, filename), compress=True)

    return features_to_accept


def get_features_importance(model):
    if isinstance(model, CatBoostRegressor) or isinstance(model, LGBMRegressor):
        return model.feature_importances_
