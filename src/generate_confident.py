import numpy as np
import pandas as pd
import xgboost
from joblib import load
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import get_current_file_parent_path
from src.data_prep.pre_process import get_features_df

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH.parent, 'data')


def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err) ** 2
        return grad, hess

    return _log_cosh_quantile


def generate_conf_plot(hparam_path, x_train, y_train, X_test, y_test):
    # sort test values
    X_test_sorted = pd.concat((X_test, y_test), axis=1).sort_values(by=y_test.name)
    y_test_sorted = X_test_sorted[y_test.name]
    X_test_sorted.drop(y_test.name, axis=1, inplace=True)

    all_models = {}
    common_params = load(hparam_path)
    common_params['n_estimators'] = int(common_params['n_estimators'])
    alpha = 0.05
    prediction_interval_percent = int((1 - 2 * alpha) * 100)
    for limit in [alpha, 1 - alpha]:
        gbr = xgboost.XGBRegressor(objective=log_cosh_quantile(limit), **common_params)
        all_models["q %1.2f" % limit] = gbr.fit(x_train, y_train)
    gbr_ls = xgboost.XGBRegressor(**common_params)

    all_models["mse"] = gbr_ls.fit(x_train, y_train)

    y_pred = all_models["mse"].predict(X_test_sorted)
    y_lower = all_models["q 0.05"].predict(X_test_sorted)
    y_upper = all_models["q 0.95"].predict(X_test_sorted)

    def draw_plot(q_lower, q_upper, real_values, save_plot=True):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(real_values, 'gx', label=u'real value')
        plt.plot(q_lower, 'b_', label=u'Q low')
        plt.plot(q_upper, 'y_', label=u'Q up')
        index = np.arange(len(q_lower))
        plt.fill(np.concatenate([index, index[::-1]]),
                 np.concatenate([q_lower, q_upper[::-1]]),
                 alpha=.25, fc='b', ec='None', label=f'{prediction_interval_percent}% prediction interval')
        # plt.plot(real_values, q_upper, "k-")
        # plt.plot(real_values, q_lower, "k-")
        # plt.fill_between(
        #     real_values, q_lower, q_upper, alpha=0.25, label=f'Predicted {prediction_interval_percent}% interval'
        # )
        plt.grid()
        plt.plot(real_values, y_pred, 'o', label='Predicted mean')

        # plt.xlabel("y_pred")
        plt.xlabel('$index$')
        # plt.ylabel("y_test")
        plt.ylabel('$PCN$')
        plt.legend(loc="upper left")
        plt.title(f'confident score for RNA{rna_type}')
        plt.show()
        if save_plot:
            fig.savefig(f'{DATA_PATH}\\conf_score_RNA{rna_type}_XGBoost.png')

    draw_plot(y_lower, y_upper, y_test_sorted.ravel())


if __name__ == '__main__':
    rna_type = 'p'
    hparam_path = Path(DATA_PATH, f'RNA{rna_type}_best_params_XGBoost.joblib')
    selected_features = load(Path(DATA_PATH, f'RNA{rna_type}_XGBoost_Selected_Features.joblib'))['selected_features']

    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train'][selected_features]
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val'][selected_features]
    RNA_y_val = data[f'RNA{rna_type}_y_val']

    RNA_X_test_features = data[f'RNA{rna_type}_X_test'][selected_features]
    RNA_y_test = data[f'RNA{rna_type}_y_test']

    x_train = pd.concat([RNA_X_train_features, RNA_X_val_features], axis=0)
    y_train = pd.concat([RNA_y_train, RNA_y_val], axis=0)
    X_test = RNA_X_test_features
    y_test = RNA_y_test

    generate_conf_plot(hparam_path, x_train, y_train, X_test, y_test)
