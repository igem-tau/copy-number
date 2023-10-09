import re
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from src.models.Features_Models_Selection import get_hyper_parameters
from src.models.models_functions import prepare_model_data
from src.data_prep.pre_process import train_validation_split
import warnings
import xgboost as xgb
from catboost import CatBoostRegressor
from xgboost.callback import EarlyStopping
import optuna
from optuna.visualization import *
from src.consts import RANDOM_STATE
from src.utils import get_current_file_parent_path, get_current_date
from joblib import dump, load
from scipy.stats import pearsonr
import plotly as py

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', '..', 'data')

warnings.simplefilter(action='ignore', category=FutureWarning)


def write_to_xl(dic, model_name):
    df1 = pd.DataFrame(dic.values())
    df1['scores'] = dic.keys()
    xl_name = f'{model_name}_best_params.xlsx'
    target_file = Path(DATA_PATH, xl_name)
    try:
        df2 = pd.read_excel(target_file)
    except:
        print(f'create xl {xl_name}')
        workbook = openpyxl.Workbook()
        workbook.save(target_file)
        df2 = pd.read_excel(target_file)
    df = pd.concat([df2, df1], ignore_index=True)
    df.to_excel(target_file, index=False)


def converge_randomsearch(X_train, X_test, y_train, y_test, dataset_name, num_of_steps=5, nun_iter=7):
    xgb_tuned = xgb.XGBRegressor(random_state=1)
    parameters_base = {'learning_rate': [0.5 / 2], 'n_estimators': [int(2000 / 2)], 'max_depth': [int(20 / 2)],
                       'gamma': [0.8 / 2], 'subsample': [0.99 / 2], 'colsample_bytree': [0.99 / 2]}

    def calc_range(key, min_val, max_val, t, parameters_base=parameters_base, ):
        [lower_n, upper_num] = [max(min_val, parameters_base[key][0] - (parameters_base[key][0] / (t + 0.5))),
                                min(max_val, parameters_base[key][0] + (parameters_base[key][0] / (t + 0.5)))]
        return ([lower_n, upper_num])

    scores = [-100]
    param_d = {}

    for step in range(num_of_steps):
        parameters_min = {'learning_rate': calc_range('learning_rate', 0.0001, 0.5, step)[0],
                          'n_estimators': calc_range('n_estimators', 2, 5000, step)[0],
                          'max_depth': calc_range('max_depth', 1, 100, step)[0],
                          'gamma': calc_range('gamma', 0, 1, step)[0],
                          'subsample': calc_range('subsample', 0, 1, step)[0],
                          'colsample_bytree': calc_range('colsample_bytree', 0.0001, 1, step)[0]}
        parameters_max = {'learning_rate': calc_range('learning_rate', 0.0001, 0.5, step)[1],
                          'n_estimators': calc_range('n_estimators', 2, 5000, step)[1],
                          'max_depth': calc_range('max_depth', 1, 100, step)[1],
                          'gamma': calc_range('gamma', 0, 1, step)[1],
                          'subsample': calc_range('subsample', 0, 1, step)[1],
                          'colsample_bytree': calc_range('colsample_bytree', 0.0001, 1, step)[1]}
        parameters = {}
        keys_pairs = [('learning_rate', 'max_depth'), ('subsample', 'n_estimators'), ('gamma', 'colsample_bytree')]
        for t in keys_pairs:
            parameters = parameters_base
            parameters[t[0]] = np.linspace(parameters_min[t[0]], parameters_max[t[0]], 4)
            if t[1] == 'max_depth' or t[1] == 'n_estimators':
                parameters[t[1]] = np.linspace(parameters_min[t[1]], parameters_max[t[1]], 4, dtype=int)
            else:
                parameters[t[1]] = np.linspace(parameters_min[t[1]], parameters_max[t[1]], 4)
            scorer = metrics.make_scorer(metrics.r2_score)
            rand_obj = RandomizedSearchCV(xgb_tuned, parameters, scoring=scorer, n_iter=nun_iter, n_jobs=-1, cv=2,
                                          verbose=1)
            rand_obj = rand_obj.fit(X_train, y_train)
            parameters_base[t[0]] = rand_obj.best_params_[t[0]]
            parameters_base[t[1]] = rand_obj.best_params_[t[1]]
            best_model = rand_obj.best_estimator_
            y_pred = best_model.predict(X_test)
            score = metrics.r2_score(y_test, y_pred)
            scores.append(score)
            parameters_base[t[0]] = [rand_obj.best_params_[t[0]]]
            parameters_base[t[1]] = [rand_obj.best_params_[t[1]]]
            print(
                f'{t[0]} was set to {rand_obj.best_params_[t[0]]} and {t[1]} was set to {rand_obj.best_params_[t[1]]}')
            if score >= max(scores):
                param_d = {score: rand_obj.best_params_}
    print(scores)
    # plt.plot(range(len(scores)), scores, label='score')
    # plt.legend()
    # plt.show()

    write_to_xl(param_d, dataset_name)
    return (scores[-1], rand_obj.best_params_)


def get_best_params_set_xgb(X_train, X_val, y_train, y_val, model_name):
    xl_name = f'{model_name}_best_params.xlsx'
    if not Path(DATA_PATH, xl_name).exists():
        for i in range(5):
            [ii, kk] = converge_randomsearch(X_train, X_val, y_train, y_val, model_name, num_of_steps=7, nun_iter=7)

    df = pd.read_excel(xl_name)
    score = df['scores'].max()
    params = df.iloc[df['scores'].idxmax(), :].dropna().drop('scores')
    print(f'Best params for {model_name} model are:\n{params}\nAnd their predicted score is {score}')
    best_params = dict(params)
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    return best_params


def find_optimal_alpha_Lasso(X, y, model_name):
    xl_name = f'{model_name}_best_params.xlsx'
    if not Path(DATA_PATH, xl_name).exists():
        X_train, X_test, y_train, y_test = prepare_model_data(X, y)
        # create a LassoCV object with 10-fold cross-validation
        print('running LassoCV to find the optimal alpha')
        lasso_cv = LassoCV(cv=10, max_iter=5000)
        # fit the Lasso model
        lasso_cv.fit(X_train, y_train)
        y_pred = lasso_cv.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        param_d = {r2: {'alpha': lasso_cv.alpha_}}
        write_to_xl(param_d, model_name)

    df = pd.read_excel(xl_name)
    score = df['scores'].max()
    params = df.iloc[df['scores'].idxmax(), :].dropna().drop('scores')
    print(f'Best params for {model_name} model are:\n{params}\nAnd their predicted score is {score}')
    return (dict(params))


def objective(trial, X_train, X_val, y_train, y_val, model_name):
    regressor_obj, param = get_hyper_parameters(trial=None, regressor_name=model_name)

    if model_name == 'XGBoost':
        es = EarlyStopping(
            rounds=30,
            data_name='validation_0',
            metric_name=scoring_function.__name__,
            maximize=True,
            save_best=True,
            min_delta=0
        )

        param['eval_metric'] = scoring_function
        param['callbacks'] = [es]

        regressor_obj = xgb.XGBRegressor(**param, random_state=RANDOM_STATE)
        regressor_obj.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr('callbacks', regressor_obj.best_iteration + 1)
        y_pred = regressor_obj.predict(X_val)

    elif model_name == 'CatBoostRegressor':
        regressor_obj.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = regressor_obj.predict(X_val)

    elif model_name == 'RandomForest':
        regressor_obj.fit(X_train, y_train)
        y_pred = regressor_obj.predict(X_val)

    elif model_name == 'LGBMRegressor':
        X_train = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_val = X_val.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        regressor_obj.fit(X_train, y_train)
        y_pred = regressor_obj.predict(X_val)

    else:
        raise ValueError(
            'hyperparameters tuning with optuna: models accepts only the following values: "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "Random Forest"')

    return r2_score(y_val, y_pred)


def scoring_function(y_true, y_predict):
    return r2_score(y_true, y_predict)


def get_best_param_optuna(X_train, X_val, y_train, y_val, model_name, rna_type, save_plots=True):
    best_params_file_name = f'RNA{rna_type}_best_params_{model_name}.joblib'
    best_params_file_path = Path(DATA_PATH, best_params_file_name)
    best_params_study_file_name = f'RNA{rna_type}_best_params_study_{model_name}.joblib'
    best_params_study_file_path = Path(DATA_PATH, best_params_study_file_name)
    # num_trials = 200
    num_trials = 1
    max_trials_per_optimization_cycle = 10

    if best_params_file_path.exists():
        best_params = load(best_params_file_path)
        return best_params

    study = optuna.create_study(direction='maximize')
    if best_params_study_file_path.exists():
        last_study = load(best_params_study_file_path)
        study.add_trials(last_study.trials)

    num_trials_left = max(0, num_trials - len(study.trials))
    print(f'Running: optuna for {model_name}, {num_trials_left} trials left')
    while num_trials_left > 0:
        current_num_trials = min(num_trials_left, max_trials_per_optimization_cycle)
        study.optimize(
            partial(objective, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, model_name=model_name),
            n_trials=current_num_trials)
        dump(study, best_params_study_file_path, compress=True)
        num_trials_left -= current_num_trials

    best_params = study.best_params
    if model_name == 'XGBoost':
        best_params['n_estimators'] = int(study.best_trial.user_attrs['callbacks'] * 1.1)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    dump(best_params, Path(DATA_PATH, best_params_file_name), compress=True)
    if save_plots:
        save_optuna_plots(study, model_name, rna_type)

    return best_params


def save_optuna_plots(study, model_name, rna_type):
    importances = optuna.importance.get_param_importances(study)
    params_sorted = list(importances.keys())

    fig1 = plot_slice(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_parallel_coordinate(study)
    fig4 = plot_timeline(study)
    fig5 = plot_rank(study, params=params_sorted[:4])
    fig6 = plot_optimization_history(study)

    with open(
            Path(DATA_PATH, f'{get_current_date()}_{model_name}_RNA{rna_type}_optuna_graphs.html'),
            'w') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig5.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig6.to_html(full_html=False, include_plotlyjs='cdn'))

## example##
# dic={1200:{'r':4,'t':300}}
# model_name='xgb'
# write_to_xl(dic,model_name)
# a=get_best_params_set(model_name)
