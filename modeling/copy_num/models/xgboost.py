from sklearn.metrics import r2_score
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt

def run_xgboost(X_train, X_test, y_train, y_test, Best_param: dict,importance_title: str = None):
    if not  bool(Best_param):
        xgb_model = xgb.XGBRegressor(Best_param)
    else:
        xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for xgboost: {r2}")

    xgb.plot_importance(xgb_model, max_num_features=20, title=importance_title)
    plt.show()
def converge_randomsearch(X_train, X_test, y_train, y_test,num_of_steps = 5,nun_iter=7):
    xgb_tuned = xgb.XGBRegressor(random_state=1)
    parameters_base = {"learning_rate": [0.5 / 2], "n_estimators": [int(2000 / 2)], "max_depth": [int(20 / 2)],
                       "gamma": [0.8 / 2], "subsample": [0.99 / 2], "colsample_bytree": [0.99 / 2]}

    def calc_range(key, min_val, max_val, t, parameters_base=parameters_base, ):
        [lower_n, upper_num] = [max(min_val, parameters_base[key][0] - (parameters_base[key][0] / (t + 0.5))),
                                min(max_val, parameters_base[key][0] + (parameters_base[key][0] / (t + 0.5)))]
        return ([lower_n, upper_num])

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
        scores = []
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
    print(scores)
    plt.plot(range(len(scores)), scores, label='score')
    plt.legend()
    plt.show()
    return(scores[-1],rand_obj.best_params_)
