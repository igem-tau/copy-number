import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from lightgbm import LGBMRegressor, plot_importance
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from src.utils import get_current_file_parent_path
from typing import Optional
import warnings
import xgboost as xgb
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import pearsonr


warnings.simplefilter(action='ignore', category=FutureWarning)

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
FIGURES_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'figures')

def run_trees_model(model_name, X_train, X_test, y_train, y_test, data_title: str = None, Best_param: Optional[dict] = None,
                save_plots: bool = False):
    if model_name == 'XGBoost':
        Best_param.pop('callbacks', None)
        Best_param['n_estimators'] = 1000
        if Best_param is not None:
            model = xgb.XGBRegressor(**Best_param)
        else:
            model = xgb.XGBRegressor()

    elif model_name == 'CatBoostRegressor':
        if Best_param is not None:
            model = CatBoostRegressor(**Best_param, allow_writing_files=False)
        else:
            model = CatBoostRegressor(allow_writing_files=False)

    elif model_name == 'RandomForest':
        if Best_param is not None:
            model = RandomForestRegressor(**Best_param)
        else:
            model = RandomForestRegressor()\

    elif model_name == 'LGBMRegressor':
        if Best_param is not None:
            model = LGBMRegressor(**Best_param)
        else:
            model = LGBMRegressor()

    else:
        raise ValueError(
            'models: models accepts only the following values: "XGBoost", "CatBoostRegressor", "LGBMRegressor" or "Random Forest"')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 value for {model_name}: {r2}')
    mae_score = mean_absolute_error(y_test, y_pred)
    print(f'MAE value for {model_name}: {mae_score}')
    pearson, _ = pearsonr(y_test, y_pred)
    print(f'pearson correlation value for {model_name}: {pearson}')
    spearman, _ = spearmanr(y_test, y_pred)
    print(f'spearman correlation value for {model_name}: {spearman}')

    if save_plots:
        if model_name == 'XGBoost':
            # Get feature importance from the XGBoost model
            importance_dict = model.get_booster().get_fscore()
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Extract the top 20 features and their importance scores
            top_features = [x[0] for x in sorted_importance[:20]]
            top_importance = [x[1] for x in sorted_importance[:20]]

            # Create a bar chart
            fig = px.bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                labels={'x': 'Feature Importance', 'y': 'Feature'},
                title=f'XGBoost Feature Importance {data_title}',
            )

            # Update the figure layout
            fig.update_layout(width=800, height=400)
            with open(Path(FIGURES_PATH, f'XGBoost feature importance {data_title}.html'), 'w') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


        elif model_name =='CatBoostRegressor' or model_name == 'RandomForest':
            feature_importance = model.feature_importances_
            sorted_idx = np.flip(np.argsort(feature_importance))
            sorted_features = np.array(X_test.columns)[sorted_idx]
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


        elif model_name =='LGBMRegressor':
            importance_dict = model.feature_importances_
            sorted_importance = sorted(enumerate(importance_dict), key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in sorted_importance[:20]]
            top_features = [X_test.columns[i] for i in top_indices]
            top_importance = [x[1] for x in sorted_importance[:20]]
            fig = px.bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                labels={'x': 'Feature Importance', 'y': 'Feature'},
                title=f'LGBMRegressor Feature Importance {data_title}',
            )
            fig.update_layout(width=800, height=400)
            with open(Path(FIGURES_PATH, f'LGBMRegressor Feature Importance {data_title}.html'), 'w') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


        # evaluation plot
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual values (log scale)', 'y': 'Predicted values (log scale)'})
        fig.add_trace(
            go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='', showlegend=False))
        fig.add_annotation(x=1, y=0.01, text=f'spearman correlation={spearman:.4f}', showarrow=False, xref='paper',
                           yref='paper', font=dict(size=15))
        fig.add_annotation(x=1, y=0.1, text=f'pearson correlation={pearson:.4f}', showarrow=False, xref='paper',
                           yref='paper' ,font=dict(size=15))
        fig.add_annotation(x=1, y=0.2, text=f'MAE={mae_score:.4f}', showarrow=False, xref='paper', yref='paper',font=dict(size=15))
        fig.update_layout(title=f'{model_name} - {data_title}')
        with open(Path(FIGURES_PATH, f'{model_name} evaluation {data_title}.html'), 'w') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    return model, r2, mae_score, pearson, spearman, y_pred