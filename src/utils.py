import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


def get_current_file_parent_path(file) -> Path:
    return Path(file).parent.resolve()


def is_feature_selected(feature: str, selected_features: 'Optional[List[str]]') -> bool:
    return selected_features is None or feature in selected_features


def get_current_date() -> str:
    return str(pd.to_datetime("today")).split()[0]


def get_estimator_features_name(estimator):
    if isinstance(estimator, CatBoostRegressor):
        return estimator.feature_names_
    elif isinstance(estimator, LGBMRegressor):
        return estimator.feature_name_
    elif isinstance(estimator, (XGBRegressor, RandomForestRegressor)):
        return estimator.feature_names_in_
    else:
        # hoping it will work, but it might break, since not all models has this attribute
        return estimator.feature_names_in_


def estimate_pred(y_true, y_pred, model_name, data_title='', estimator=None, save_plots=True, loglog_axes=True):
    FIGURES_PATH = Path(get_current_file_parent_path(__file__).parent, 'data', 'figures')

    r2 = r2_score(y_true, y_pred)
    print(f'R^2 value for {model_name}: {r2}')
    mae_score = mean_absolute_error(y_true, y_pred)
    print(f'MAE value for {model_name}: {mae_score}')
    pearson, pearson_p_value = pearsonr(y_true, y_pred)
    print(f'pearson correlation value for {model_name}: {pearson}')
    spearman, spearman_p_value = spearmanr(y_true, y_pred)
    print(f'spearman correlation value for {model_name}: {spearman}')

    if save_plots:
        if estimator is not None:
            # feature importance
            feature_importance = estimator.feature_importances_
            sorted_idx = np.flip(np.argsort(feature_importance))
            sorted_features = np.array(get_estimator_features_name(estimator))[sorted_idx]
            sorted_importance = feature_importance[sorted_idx]
            fig = px.bar(
                x=sorted_importance,
                y=sorted_features,
                orientation='h',
                labels={'x': 'Feature Importance', 'y': 'Feature'},
                title=f'{model_name} Feature Importance {data_title}',
            )
            fig.update_layout(width=800, height=400)
            with open(Path(FIGURES_PATH, f'{model_name} Feature Importance {data_title}.html'), 'w') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # evaluation plot
        fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual Copy Number', 'y': 'Predicted Copy Number'})
        fig.add_trace(
            go.Scatter(x=[max(0, min([*y_true, *y_pred])), max([*y_true, *y_pred])],
                       y=[max(0, min([*y_true, *y_pred])), max([*y_true, *y_pred])],
                       mode='lines', name='',
                       showlegend=False))
        fig.add_annotation(x=1, y=0.01, text=f'Spearman correlation={spearman:.4f}, p-value={spearman_p_value:.2e}', showarrow=False, xref='paper',
                           yref='paper', font=dict(size=15))
        fig.add_annotation(x=1, y=0.1, text=f'Pearson correlation={pearson:.4f}, p-value={pearson_p_value:.2e}', showarrow=False, xref='paper',
                           yref='paper', font=dict(size=15))
        fig.add_annotation(x=1, y=0.2, text=f'Mean Absolut Error={mae_score:.4f}', showarrow=False, xref='paper', yref='paper',
                           font=dict(size=15), align='right')
        fig.update_layout(title=f'{model_name} evaluation for {data_title}')

        if loglog_axes:
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=0.5, tickformat='.1f', type='log'),
                yaxis=dict(tickmode='linear', dtick=0.5, tickformat='.1f', type='log')
            )

        with open(Path(FIGURES_PATH, f'{model_name} evaluation {data_title}.html'), 'w') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return r2, mae_score, pearson, spearman


if __name__ == '__main__':
    # print(f'the current file parent path is: {get_current_file_parent_path(__file__)}')
    # test_features = ["TTT__TC_count", "A_count", "gc_skew", "z_curve_y",
    #                  "ada", "fhlA", "Fis_26-48", "UxuR_14-2", "pssm_score",
    #                  "Predicted Promoter Strength (KbT)",
    #                  "TCMCTCCTTT", "CGCGTTWG", "WNGCNCTYYT",
    #                  "(-11, -8) predicted strength",
    #                  'G_-35', 'T_-30', 'A_-19', 'C_-2'
    #                  ]
    # write_selected_features(test_features)
    pass
