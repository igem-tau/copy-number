from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from src.utils import get_current_file_parent_path, get_current_date
from pathlib import Path
import plotly.express as px

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


def plot_features_dist(df):
    features = df.columns
    fig = make_subplots(rows=2, cols=5, subplot_titles=features)
    for i, feature in enumerate(features):
        if feature == 'Fis_26-25':
            cur_data = df[feature][df[feature] != 1]
        else:
            cur_data = df[feature]
        if feature < 5:
            fig.add_trace(go.Histogram(x=cur_data, name=feature), row=1, col=i + 1)
        else:
            fig.add_trace(go.Histogram(x=cur_data, name=feature), row=2, col=i - 4)
    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Counts", row=2, col=1)
    fig.update_layout(template='plotly_white', title_text='<b>Features Distributions', title_x=0.5)
    return fig


def plot_features_box(df):
    numeric_features = df[['Fis_26-25', 'at_skew', 'dG_total']]
    fig = go.Figure()
    for col in numeric_features.columns:
        if col == 'Fis_26-25':
            cur_data = numeric_features[col][numeric_features[col] != 1]
        else:
            cur_data = numeric_features[col]
        fig.add_trace(go.Box(y=cur_data, boxpoints='all', name=col))

    fig.update_layout(template='plotly_white', title='<b>Features Boxplots', title_x=0.5,
                      yaxis_title_text='Features values')
    return fig


def plot_scatter_hist(df_x, df_y):
    fig = make_subplots(rows=2, cols=5, subplot_titles=df_x.columns)
    numeric_features = df_x[['Fis_26-25', 'at_skew', 'dG_total']]
    discrete_features = df_x[[col for col in df_x.columns if col not in numeric_features.columns]]

    for i, col in enumerate(df_x.columns):
        if col in discrete_features:
            plot = px.violin(x=df_x[col], y=df_y)
        else:
            if col == 'Fis_26-25':
                cur_data = df_x[col][df_x[col] != 1]
                cur_df_y = df_y[df_x[col] != 1]
            else:
                cur_data = df_x[col]
                cur_df_y = df_y
            plot = px.scatter(
                x=cur_data,
                y=cur_df_y)

        for trace in plot['data']:
            if i < 5:
                fig.add_trace(trace, row=1, col=i + 1)
            else:
                fig.add_trace(trace, row=2, col=i - 4)
    fig.update_yaxes(title_text="Copy Number", row=1, col=1)
    fig.update_yaxes(title_text="Copy Number", row=2, col=1)
    fig.update_layout(template='plotly_white', title='<b>Features vs. Copy number', title_x=0.5)

    return fig


def exploratory_data_analysis(train: pd.DataFrame, val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
                              rna_type: str) -> None:
    df_x = pd.concat([train, val])
    df_y = pd.concat([y_train, y_val])
    fig1 = plot_features_dist(df_x)
    fig2 = plot_features_box(df_x)
    fig3 = plot_scatter_hist(df_x, df_y)

    eda_save_path = Path(DATA_PATH, f'{get_current_date()}_RNA{rna_type}_features_graphs.html')
    with open(eda_save_path, 'w') as f:
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
