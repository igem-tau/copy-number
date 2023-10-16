import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from joblib import load
from pathlib import Path
from src.consts import RANDOM_STATE
from src.utils import get_current_file_parent_path
from catboost import CatBoostRegressor

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


def generate_conf_plot(hparam_path, x_train, y_train, X_test, y_test, model_name='CatBoostRegressor'):
    # sort test values
    X_test_sorted = pd.concat((X_test, y_test), axis=1).sort_values(by=y_test.name)
    y_test_sorted = X_test_sorted[y_test.name]
    X_test_sorted.drop(y_test.name, axis=1, inplace=True)

    all_models = {}
    common_params = load(hparam_path)
    alpha = 0.05
    prediction_interval_percent = int((1 - 2 * alpha) * 100)
    if model_name == 'XGBoost':
        common_params['n_estimators'] = int(common_params['n_estimators'])
        for limit in [alpha, 1 - alpha]:
            gbr = XGBRegressor(objective=log_cosh_quantile(limit), **common_params, random_state=RANDOM_STATE)
            all_models["q %1.2f" % limit] = gbr.fit(x_train, y_train)
        gbr_ls = XGBRegressor(**common_params)
    elif model_name == 'CatBoostRegressor':
        common_params.pop('loss_function', None)
        for limit in [alpha, 1 - alpha]:
            gbr = CatBoostRegressor(loss_function=f'Quantile:alpha={limit}', **common_params, random_state=RANDOM_STATE)
            all_models["q %1.2f" % limit] = gbr.fit(x_train, y_train)
        gbr_ls = CatBoostRegressor(**common_params)
    else:
        raise 'MODEL_NAME NOT SUPPORTED'

    all_models["mse"] = gbr_ls.fit(x_train, y_train)

    y_pred = all_models["mse"].predict(X_test_sorted)
    y_lower = all_models[f"q {alpha}"].predict(X_test_sorted)
    y_upper = all_models[f"q {1 - alpha}"].predict(X_test_sorted)

    def draw_plot(q_lower, q_upper, real_values, save_plot=True):
        accu = 100 * len(real_values[(q_upper - real_values > 0) * (y_pred - q_lower > 0)]) / len(real_values)
        import plotly.graph_objects as go

        fig = go.Figure()
        index = np.arange(len(q_lower))
        fig.add_trace(go.Scatter(x=index, y=q_lower,
                                 fill=None,
                                 mode='lines',
                                 line_color='#005eb8',
                                 name='Q low'
                                 ))
        fig.add_trace(go.Scatter(
            x=index,
            y=q_upper,
            fill='tonexty',  # fill area between trace0 and trace1
            mode='lines', line_color='#005eb8', fillcolor='rgba(0, 95, 184, 0.7)',
            name=f'Q up, with {prediction_interval_percent}% prediction interval'))
        fig.add_trace(go.Scatter(
            x=index,
            y=real_values,
            mode='markers', line_color='#21d19f', name='real value'))
        fig.add_annotation(x=0.1, y=0.9, text=f'Accuracy={accu:.3f}%', showarrow=False, xref='paper', yref='paper',
                           font=dict(size=15), align='right')
        fig.update_layout(title=f'Confidence Score for RNA{" ".join(rna_type.split("_"))}',
                          xaxis_title='index', yaxis_title='Copy Number')
        fig.show()

        if save_plot:
            fig.write_html(Path(DATA_PATH, 'figures', f'conf_score_RNA{rna_type}.html'), full_html=False,
                   include_plotlyjs='cdn')

    draw_plot(y_lower, y_upper, y_test_sorted.ravel())


if __name__ == '__main__':
    # TODO - Update the rna_type before running to match your data
    rna_type = 'p_fitted'
    model_name = 'CatBoostRegressor'
    hparam_path = Path(DATA_PATH, f'RNA{rna_type}_best_params_{model_name}.joblib')
    selected_features = load(Path(DATA_PATH, f'RNA{rna_type}_{model_name}_Selected_Features.joblib'))

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

    generate_conf_plot(hparam_path, x_train, y_train, X_test, y_test, model_name)
