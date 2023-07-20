from src.data_prep.pre_process import get_features_df
from src.models.models_functions import model
import matplotlib.pyplot as plt
import os

FIGURES_PATH = os.path.join('..', '..', 'data', 'figures')


def elbow_graphs(estimator, estimator_range, X, y, data_name):
    r2_list = []
    sp_list = []
    for estimator_value in estimator_range:
        r2, mse_score, spearman = model(X, y, model_name='xgboost', data_name=data_name,
                              Best_param={estimator: estimator_value}, save_plots=False)
        r2_list.append(r2)
        sp_list.append(spearman)

    # plot
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('number of trees' if estimator=='n_estimators' else 'maximum depth of a tree')
    ax1.set_ylabel('r2 score', color=color)
    ax1.plot(estimator_range, r2_list, color=color)
    ax1.scatter(estimator_range, r2_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('spearman', color=color)  # we already handled the x-label with ax1
    ax2.plot(estimator_range, sp_list)
    ax2.scatter(estimator_range, sp_list)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'XGBoost {estimator} vs. scores for {data_name}')
    fig.tight_layout()

    plt.savefig(os.path.join(FIGURES_PATH, f'XGBoost {estimator} vs scores {data_name}.jpg'))


if __name__ == '__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    elbow_graphs('n_estimators', range(10, 300, 10), RNAp_X, RNAp_y, 'RNAp')
    elbow_graphs('max_depth', range(2, 20, 1), RNAp_X, RNAp_y, 'RNAp')

    elbow_graphs('n_estimators', range(10, 300, 10), RNAi_X, RNAi_y, 'RNAi')
    elbow_graphs('max_depth', range(2, 20, 1), RNAi_X, RNAi_y, 'RNAi')

    elbow_graphs('n_estimators', range(10, 300, 10), X_shared_model, Y_shared_model, 'shared model')
    elbow_graphs('max_depth', range(2, 20, 1), X_shared_model, Y_shared_model, 'shared model')
