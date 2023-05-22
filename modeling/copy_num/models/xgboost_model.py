import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional
import warnings
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)

FIGURES_PATH = os.path.join("..", "..", "..", "data", "copy_num", "figures")


def run_xgboost(X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if Best_param is not None:
        Best_param['max_depth'], Best_param['n_estimators'] = int(Best_param['max_depth']), int(Best_param['n_estimators'])
        xgb_model = xgb.XGBRegressor(**Best_param)
    else:
        xgb_model = xgb.XGBRegressor()

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for xgboost: {r2}")
    mse_score = mean_squared_error(y_test, y_pred)
    print('the mse score for xgboost %.5f' % mse_score)
    spearman, _ = spearmanr(y_test, y_pred)
    print(f"spearman correlation value for xgboost: {spearman}")

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

    return r2, mse_score, spearman
