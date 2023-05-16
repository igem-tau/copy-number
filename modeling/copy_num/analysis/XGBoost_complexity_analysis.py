from modeling.copy_num.data_prep.pre_process import get_features_df
from modeling.copy_num.models.models_functions import model
import matplotlib.pyplot as plt
import os

FIGURES_PATH = os.path.join("..", "..", "..", "data", "copy_num", "figures")

if __name__ == '__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    n_estimators_range = range(4, 1000, 20)
    r2_list = []
    mse_list = []
    for n_estimators in n_estimators_range:
        r2, mse_score = model(RNAp_X, RNAp_y, model_name="xgboost", data_name="pRNA",
                              Best_param={'n_estimators': n_estimators}, save_plots=False)
        r2_list.append(r2)
        mse_list.append(mse_score)

    # plot
    fig = plt.figure()
    plt.errorbar(n_estimators_range, r2_list)
    # plt.errorbar(n_estimators_range, mse_list)

    plt.title("XGBoost n_estimators vs R2 score")
    plt.xlabel('n_estimators')
    plt.ylabel('R2')
    plt.savefig(os.path.join(FIGURES_PATH, f'XGBoost n_estimators vs R2 score pRNA.jpg'))
