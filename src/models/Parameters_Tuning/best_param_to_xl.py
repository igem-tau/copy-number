import numpy as np
import openpyxl
import os
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.models.models_functions import prepare_model_data
from src.data_prep.pre_process import train_validation_split
import warnings
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)


def write_to_xl(dic, model_name):
    df1=pd.DataFrame(dic.values())
    df1['scores']=dic.keys()
    xl_name=f'{model_name}_best_params.xlsx'
    target_file=os.path.join(os.getcwd(), xl_name)
    try:
        df2 = pd.read_excel(target_file)
    except:
        print(f'create xl {xl_name}')
        workbook = openpyxl.Workbook()
        workbook.save(target_file)
        df2=pd.read_excel(target_file)
    df= pd.concat([df2, df1], ignore_index=True)
    df.to_excel(target_file,index=False)



def converge_randomsearch(X_train, X_test, y_train, y_test,dataset_name, num_of_steps = 5,nun_iter=7):
    xgb_tuned = xgb.XGBRegressor(random_state=1)
    parameters_base = {'learning_rate': [0.5 / 2], 'n_estimators': [int(2000 / 2)], 'max_depth': [int(20 / 2)],
                       'gamma': [0.8 / 2], 'subsample': [0.99 / 2], 'colsample_bytree': [0.99 / 2]}

    def calc_range(key, min_val, max_val, t, parameters_base=parameters_base, ):
        [lower_n, upper_num] = [max(min_val, parameters_base[key][0] - (parameters_base[key][0] / (t + 0.5))),
                                min(max_val, parameters_base[key][0] + (parameters_base[key][0] / (t + 0.5)))]
        return ([lower_n, upper_num])

    scores = [-100]
    param_d={}

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
            rand_obj = RandomizedSearchCV(xgb_tuned, parameters, scoring=scorer, n_iter=nun_iter, n_jobs=-1, cv=2, verbose=1)
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
            if score>=max(scores):
                param_d={score:rand_obj.best_params_}
    print(scores)
    # plt.plot(range(len(scores)), scores, label='score')
    # plt.legend()
    # plt.show()

    write_to_xl(param_d, dataset_name)
    return(scores[-1],rand_obj.best_params_)


def get_best_params_set_xgb(X_train, X_val, y_train, y_val, model_name, stratify_by):
    xl_name = f'{model_name}_best_params.xlsx'
    if not os.path.exists(os.path.join(os.getcwd(), xl_name)):
        for i in range(5):
            [ii, kk] = converge_randomsearch(X_train, X_val, y_train, y_val, model_name, num_of_steps=7, nun_iter=7)

    df = pd.read_excel(xl_name)
    score = df['scores'].max()
    params = df.iloc[df['scores'].idxmax(),:].dropna().drop('scores')
    print(f'Best params for {model_name} model are:\n{params}\nAnd their predicted score is {score}')
    return (dict(params))


def find_optimal_alpha_Lasso(X, y, model_name):
    xl_name = f'{model_name}_best_params.xlsx'
    if not os.path.exists(os.path.join(os.getcwd(), xl_name)):
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

## example##
# dic={1200:{'r':4,'t':300}}
# model_name='xgb'
# write_to_xl(dic,model_name)
# a=get_best_params_set(model_name)






