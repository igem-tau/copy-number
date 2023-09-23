from joblib import dump
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_prep.pre_process import get_features_df, train_validation_split
from src.models.models_functions import remove_outliers
from src.utils import get_current_file_parent_path
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from tqdm import tqdm
from typing import Any, Dict, Literal, Union
from xgboost import XGBRegressor


RNA_TYPES = Literal['p', 'i']
RELATIVE_DATA_PATH = Path('..', '..', 'data', 'feature_selection')


def get_seeds(num_seeds: int) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(low=0, high=2**32, size=num_seeds)


def get_save_path(rna_type: RNA_TYPES, model_type: str, file_type: str = 'joblib') -> Path:
    return Path(
        get_current_file_parent_path(__file__),
        RELATIVE_DATA_PATH,
        f'{model_type}_features_selection_RNA{rna_type}.'+file_type
    )


def forward_selection(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, model: Any):
    features = list(X_train.columns)
    selected_features = []
    best_r2_score = None

    while True:
        r2_scores = []
        for current_feature in features:
            model.fit(X_train[selected_features + [current_feature]], y_train)

            y_pred = model.predict(X_valid[selected_features + [current_feature]])
            r2_scores.append(r2_score(y_valid, y_pred))

        best_features_index = np.array(r2_scores).argmax()
        new_best_r2_score = r2_scores[best_features_index]
        if best_r2_score is None or new_best_r2_score > best_r2_score:
            best_r2_score = new_best_r2_score
            best_feature = features[best_features_index]
            selected_features.append(best_feature)
            features.remove(best_feature)
        else:
            break

    return selected_features, best_r2_score


def multi_split_forward_selection(data: Dict[str, Union[pd.DataFrame, pd.Series]], rna_type: RNA_TYPES,
                                  model: Any, model_type:  str) -> pd.DataFrame:
    X_train_val, y_train_val = data[f'RNA{rna_type}_X_train_val'], data[f'RNA{rna_type}_y_train_val']
    rows = []
    # X, y = remove_outliers(X, y)
    seeds = get_seeds(20)
    for seed in tqdm(seeds):
        X_train, X_val, y_train, y_val = train_validation_split(
            X_train_val, y_train_val,
            stratify_by=data[f'RNA{rna_type}_stratify_by'], random_state=seed
        )
        X_test = data[f'RNA{rna_type}_X_test']
        y_test = data[f'RNA{rna_type}_y_test']

        selected_features, validation_score = forward_selection(X_train, y_train, X_val, y_val, model)
        model.fit(X_train_val[selected_features], y_train_val)
        y_pred = model.predict(X_test[selected_features])
        test_score = r2_score(y_test, y_pred)
        new_row = pd.Series({'features': selected_features, 'validation_score': validation_score,
                             'test_score': test_score})
        rows.append(new_row)

    results = pd.concat(rows, axis=1).T
    dump(results, get_save_path(rna_type, model_type, file_type='joblib'), compress=True)
    results.to_csv(get_save_path(rna_type, model_type, file_type='csv'), index=False)
    return results


def main():
    data = get_features_df()
    RNA_TYPE_RUNS = ('p', 'i')
    MODEL_RUNS = (
        (LinearRegression(), 'linear_regression'),
        (XGBRegressor(), 'xgboost')
    )

    for rna_type in RNA_TYPE_RUNS:
        for model, model_type in MODEL_RUNS:
            print(f'start multi forward selection of RNA type: {rna_type}, with model: {model_type}')
            multi_split_forward_selection(data, rna_type, model, model_type)


if __name__ == '__main__':
    main()
