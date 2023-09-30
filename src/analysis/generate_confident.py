import numpy as np
import pandas as pd
import xgboost
from joblib import load
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import get_current_file_parent_path
from src.data_prep.pre_process import get_features_df
CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH.parent.parent, 'data')

def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

def generate_conf_plot(hparam_path,x_train,y_train,X_test,y_test):
    all_models = {}
    common_params= load(hparam_path)
    common_params['n_estimators']=int(common_params['n_estimators'])
    for alpha in [0.05, 0.95]:
        gbr = xgboost.XGBRegressor(objective=log_cosh_quantile(alpha), **common_params)
        all_models["q %1.2f" % alpha] = gbr.fit(x_train,y_train)
    gbr_ls=xgboost.XGBRegressor(objective="reg:squarederror", **common_params)

    all_models["mse"] = gbr_ls.fit(x_train,y_train)

    y_pred = all_models["mse"].predict(X_test)
    y_lower = all_models["q 0.05"].predict(X_test)
    y_upper = all_models["q 0.95"].predict(X_test)

    fig = plt.figure(figsize=(10, 10))
    # plt.plot(y_test, y_upper, "k-")
    # plt.plot(y_test, y_lower, "k-")
    plt.fill_between(
        y_test.ravel(), y_lower, y_upper, alpha=0.25, label="Predicted 90% interval"
    )
    plt.plot(y_test, y_pred,'o' ,label="Predicted mean")

    plt.xlabel("y_pred")
    plt.ylabel("y_test")
    plt.legend(loc="upper left")
    plt.title(f'confident socre for RNA{rna_type}')
    plt.show()
    fig.savefig(f'{DATA_PATH}\\conf_score_RNA{rna_type}_{model_name}.png')

if __name__=='__main__':
    rna_type='p'
    hparam_path=Path(DATA_PATH, f'best_params_XGBoost.joblib')
    selected_features=load(Path(DATA_PATH, f'RNA{rna_type}_Selected_Features.joblib'))['XGBoost']['selected_features']

    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train'][selected_features]
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val'][selected_features]
    RNA_y_val = data[f'RNA{rna_type}_y_val']

    RNA_X_test_features = data[f'RNA{rna_type}_X_test'][selected_features]
    RNA_y_test = data[f'RNA{rna_type}_y_test']

    x_train=pd.concat([RNA_X_train_features,RNA_X_val_features],axis=0)
    y_train=pd.concat([RNA_y_train,RNA_y_val],axis=0)
    X_test=RNA_X_test_features
    y_test=RNA_y_test

    generate_conf_plot(hparam_path,x_train,y_train,X_test,y_test)
