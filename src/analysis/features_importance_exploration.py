from joblib import dump, load
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from src.consts import RANDOM_STATE
from src.data_prep.pre_process import get_features_df
from src.models.models_functions import prepare_model_data
from src.utils import get_current_file_parent_path
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from tqdm import tqdm
from typing import Literal
from xgboost import XGBRegressor


NUM_RUNS = 5000
RNA_TYPES = Literal['p', 'i']
MODEL_TYPES = Literal['lasso', 'xgboost']
RELATIVE_DATA_PATH = Path('..', '..', 'data')
RELATIVE_FIGURES_PATH = Path(RELATIVE_DATA_PATH, 'figures')


def get_seeds(num_seeds: int) -> np.ndarray:
    return np.random.randint(low=0, high=2 ** 32 - 1, size=num_seeds)


def get_features_exploration_path(model_type: MODEL_TYPES, rna_type: RNA_TYPES) -> Path:
    return Path(
        get_current_file_parent_path(__file__),
        RELATIVE_DATA_PATH,
        f'{model_type}_features_importance_RNA{rna_type}_exploration_data.joblib'
    )


def get_features_exploration_plot_path(model_type: MODEL_TYPES, rna_type: RNA_TYPES) -> Path:
    return Path(
        get_current_file_parent_path(__file__),
        RELATIVE_FIGURES_PATH,
        f'{model_type}_RNA{rna_type}_features_importance_exploration.jpg'
    )


def get_features_exploration_data(X: pd.DataFrame, y: 'pd.Series[int]', model_type: MODEL_TYPES, rna_type: RNA_TYPES) -> pd.DataFrame:
    file_path = get_features_exploration_path(model_type, rna_type)

    if Path(file_path).exists() and Path(file_path).is_file():
        return load(file_path)

    seed_array = get_seeds(NUM_RUNS)
    columns = [*X.columns, 'R2 Score']
    run_rows = []

    for seed in tqdm(np.unique(seed_array)):
        X_train, X_test, y_train, y_test = prepare_model_data(X, y, random_state=seed)

        if model_type == 'lasso':
            model = Lasso(alpha=0.3, max_iter=5000, random_state=RANDOM_STATE)
        elif model_type == 'xgboost':
            model = XGBRegressor(random_state=RANDOM_STATE)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        new_row = pd.DataFrame(np.append(model.coef_, score), columns).T
        run_rows.append(new_row.rename(index={0: seed}))

    runs_df = pd.concat(run_rows)
    dump(runs_df, file_path, compress=True)

    return runs_df


def plot_features_exploration(runs_df, model_type: MODEL_TYPES, rna_type: RNA_TYPES) -> None:
    NUMBER_OF_FEATURES = 55

    runs_df = runs_df.replace(0, np.nan)
    runs_features = runs_df.drop('R2 Score', axis=1)
    features_scores_average = runs_features.apply(
        lambda column: runs_df.loc[column.notna(), 'R2 Score'].mean()
    )
    features_stats = pd.concat(
        [runs_features.count(), runs_features.mean(), features_scores_average], axis=1
    ).rename(columns={0: 'Count', 1: 'Average Coef', 2: 'Average Score'})
    features_stats = features_stats.sort_values(by='Average Coef', key=np.abs, ascending=False)
    features_stats.fillna(0, inplace=True)
    top_features_stats = features_stats[:NUMBER_OF_FEATURES]

    plt.figure(figsize=(30, 7))
    color_map = 'rocket_r'
    bar = sns.barplot(data=top_features_stats, x=top_features_stats.index, y='Count', hue='Average Score',
                      dodge=False, palette=color_map)

    bar.legend_.remove()
    norm = plt.Normalize(top_features_stats['Average Score'].min(), top_features_stats['Average Score'].max())
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    plt.colorbar(sm, ax=bar, pad=0.01).set_label('R2 Average Score', rotation=90)

    plt.title(f'{model_type} RNA{rna_type} feature importance')
    plt.xticks(rotation=90)

    plt.savefig(get_features_exploration_plot_path(model_type, rna_type), bbox_inches='tight')


def main() -> None:
    data = get_features_df()
    RNAp_data = (data['RNAp_X_train_val'], data['RNAp_y_train_val'])
    RNAi_data = (data['RNAi_X_train_val'], data['RNAi_y_train_val'])

    for model_type in ['lasso', 'xgboost']:
        for (X, y), rna_type in [(RNAp_data, 'p'), (RNAi_data, 'i')]:
            runs_df = get_features_exploration_data(X, y, model_type, rna_type)
            plot_features_exploration(runs_df, model_type, rna_type)


if __name__ == '__main__':
    main()
