from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from modeling.copy_num.models.Parameters_Tuning.best_param_to_xl import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from modeling.copy_num.models.Parameters_Tuning import best_param_to_xl

FIGURES_PATH = os.path.join("..", "..", "..", "data", "copy_num", "figures")

def run_xgboost(X_train, X_test, y_train, y_test, data_title: str = None, Best_param: dict = {}, save_plots=False):
    if bool(Best_param):
        xgb_model = xgb.XGBRegressor(**Best_param)
    else:
        xgb_model = xgb.XGBRegressor()

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for xgboost: {r2}")
    mse_score = mean_squared_error(y_test, y_pred)
    print('the mse score for xgboost %.5f' % mse_score)

    ax = xgb.plot_importance(xgb_model, max_num_features=20, title=f'{data_title} feature importance - XGBoost')
    if save_plots:
        ax.figure.savefig(os.path.join(FIGURES_PATH, f'XGBoost feature importance {data_title}.jpg'))

    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_test, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'R2=%.4f' % r2, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MSE=%.4f' % mse_score, transform=ax.transAxes)
    plt.title(f'XGBoost - {data_title}')
    if save_plots:
        plt.savefig(os.path.join(FIGURES_PATH, f'XGBoost evaluation {data_title}.jpg'))

    return r2, mse_score



def converge_randomsearch(X_train, X_test, y_train, y_test,num_of_steps = 5,nun_iter=7):
    xgb_tuned = xgb.XGBRegressor(random_state=1)
    parameters_base = {"learning_rate": [0.5 / 2], "n_estimators": [int(2000 / 2)], "max_depth": [int(20 / 2)],
                       "gamma": [0.8 / 2], "subsample": [0.99 / 2], "colsample_bytree": [0.99 / 2]}

    def calc_range(key, min_val, max_val, t, parameters_base=parameters_base, ):
        [lower_n, upper_num] = [max(min_val, parameters_base[key][0] - (parameters_base[key][0] / (t + 0.5))),
                                min(max_val, parameters_base[key][0] + (parameters_base[key][0] / (t + 0.5)))]
        return ([lower_n, upper_num])

    scores = [-100]
    param_d={}

    for step in range(num_of_steps):
        parameters_min = {"learning_rate": calc_range("learning_rate", 0.0001, 0.5, step)[0],
                          "n_estimators": calc_range("n_estimators", 2, 5000, step)[0],
                          "max_depth": calc_range("max_depth", 1, 100, step)[0],
                          "gamma": calc_range("gamma", 0, 1, step)[0],
                          "subsample": calc_range("subsample", 0, 1, step)[0],
                          "colsample_bytree": calc_range("colsample_bytree", 0.0001, 1, step)[0]}
        parameters_max = {"learning_rate": calc_range("learning_rate", 0.0001, 0.5, step)[1],
                          "n_estimators": calc_range("n_estimators", 2, 5000, step)[1],
                          "max_depth": calc_range("max_depth", 1, 100, step)[1],
                          "gamma": calc_range("gamma", 0, 1, step)[1],
                          "subsample": calc_range("subsample", 0, 1, step)[1],
                          "colsample_bytree": calc_range("colsample_bytree", 0.0001, 1, step)[1]}
        parameters = {}
        keys_pairs = [("learning_rate", "max_depth"), ("subsample", "n_estimators"), ("gamma", "colsample_bytree")]
        for t in keys_pairs:
            parameters = parameters_base
            parameters[t[0]] = np.linspace(parameters_min[t[0]], parameters_max[t[0]], 4)
            if t[1] == "max_depth" or t[1] == "n_estimators":
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

    write_to_xl(param_d, 'xgb')
    return(scores[-1],rand_obj.best_params_)
