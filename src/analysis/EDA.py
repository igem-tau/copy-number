from typing import Union
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import get_current_file_parent_path, get_current_date, get_continuous_and_discrete_features
from pathlib import Path
import plotly.express as px

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


def plot_features_dist(df):
    features = df.columns
    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    for i, feature in enumerate(features):
        # TODO - understand what is so special with this feature
        if feature == 'Fis_26-25':
            cur_data = df.query(f'{feature} != 1')[feature]
        else:
            cur_data = df[feature]

        fig.add_trace(go.Histogram(x=cur_data, name=feature), row=i // n_cols + 1, col=i % n_cols + 1)

    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=1)
    fig.update_layout(template='plotly_white', title_text='<b>Features Distributions', title_x=0.5)
    return fig


def plot_features_box(df):
    # continuous_features = ['Fis_26-25', 'at_skew', 'dG_total']
    continuous_features, _ = get_continuous_and_discrete_features(df)
    fig = go.Figure()
    for col in continuous_features:
        # TODO - understand what is so special with this feature
        if col == 'Fis_26-25':
            cur_data = df.query(f'{col} != 1')[col]
        else:
            cur_data = df[col]
        fig.add_trace(go.Box(y=cur_data, boxpoints='all', name=col))

    fig.update_layout(template='plotly_white', title='<b>Features Boxplots', title_x=0.5,
                      yaxis_title_text='Features values')
    return fig


def plot_scatter_hist(df_x, df_y):
    features = df_x.columns
    # continuous_features = ['Fis_26-25', 'at_skew', 'dG_total']
    continuous_features, discrete_features = get_continuous_and_discrete_features(df_x)

    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    for i, col in enumerate(features):
        if col in discrete_features:
            plot = px.violin(x=df_x[col], y=df_y)
        else:
            # TODO - understand what is so special with this feature
            if col == 'Fis_26-25':
                cur_data = df_x.query(f'{col} != 1')[col]
                cur_df_y = df_y[df_x[col] != 1]
            else:
                cur_data = df_x[col]
                cur_df_y = df_y

            plot = px.scatter(x=cur_data, y=cur_df_y)

        for trace in plot['data']:
            fig.add_trace(trace, row=i // n_cols + 1, col=i % n_cols + 1)

    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Copy Number", row=i, col=1)

    fig.update_layout(template='plotly_white', title='<b>Features vs. Copy number', title_x=0.5)

    return fig


def exploratory_data_analysis(model_name: str, trained_model: Union[
    XGBRegressor, CatBoostRegressor, LGBMRegressor, RandomForestRegressor], train: pd.DataFrame,
                              val: pd.DataFrame, y_train: pd.Series,
                              y_val: pd.Series, rna_type: str) -> None:
    num_features_to_take = 10
    feature_importances = trained_model.feature_importances_
    importance_threshold = sorted(feature_importances, reverse=True)[:num_features_to_take][-1]
    df_x = pd.concat([train, val]).loc[:, feature_importances >= importance_threshold]
    df_y = pd.concat([y_train, y_val])
    fig1 = plot_features_dist(df_x)
    fig2 = plot_features_box(df_x)
    fig3 = plot_scatter_hist(df_x, df_y)

    eda_save_path = Path(DATA_PATH, f'{get_current_date()}_{model_name}_RNA{rna_type}_features_graphs.html')
    with open(eda_save_path, 'w') as f:
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))


def exploratory_data_analysis_by_features(features: list[str], df_x: pd.DataFrame, df_y: pd.Series,
                                          rna_type: str) -> None:
    fig1 = plot_features_dist(df_x[features])
    fig2 = plot_features_box(df_x[features])
    fig3 = plot_scatter_hist(df_x[features], df_y)

    eda_save_path = Path(DATA_PATH, f'{get_current_date()}_voting_model_RNA{rna_type}_features_graphs.html')
    with open(eda_save_path, 'w') as f:
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
