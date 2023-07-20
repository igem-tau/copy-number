from joblib import dump
from src.data_prep.pre_process import get_features_df
from src.models.models_functions import remove_outliers, train_validation_test_split
import numpy as np
import pandas as pd
import pathlib
import xgboost as xgb
from sklearn.metrics import r2_score
from tqdm import tqdm
from typing import Literal


RNA_TYPES = Literal['p', 'i']
RELATIVE_DATA_PATH = pathlib.Path('..', '..', 'data')


def get_current_file_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve()


def get_seeds(num_seeds: int) -> np.ndarray:
    return np.random.randint(low=0, high=2 ** 30, size=num_seeds)


def get_save_path(rna_type: RNA_TYPES, file_type='joblib') -> pathlib.Path:
    return pathlib.Path(
        get_current_file_path(),
        RELATIVE_DATA_PATH,
        f'new_XGB_features_selection_RNA{rna_type}.'+file_type
    )


def prediction_regressor(X_train, y_train, X_valid, y_valid):
    features = list(X_train.columns)
    selected_features = []
    best_r2_score = None

    while True:
        r2_scores = []
        for current_feature in features:
            model = xgb.XGBRegressor()
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


def main():
    data = get_features_df(p=True, i=False, shared=False, specify_date=True)
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']

    # RUNS = ((RNAp_X, RNAp_y, 'p'), (RNAi_X, RNAi_y, 'i'))
    RUNS = [(RNAp_X, RNAp_y, 'p')]

    for (X, y, rna_type) in RUNS:
        print(f'start RNA{rna_type}')
        df = pd.DataFrame([], columns=['features', 'validation score', 'test score'])
        rows = []
        X, y = remove_outliers(X, y)
        seeds = get_seeds(20)
        for seed in tqdm(seeds):
            X_train, X_valid, X_test, y_train, y_valid, y_test = train_validation_test_split(X, y, seed)
            selected_features, test_score = prediction_regressor(X_train, y_train, X_test, y_test)
            model = xgb.XGBRegressor()
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_valid[selected_features])
            validation_score = r2_score(y_valid, y_pred)
            new_row = pd.Series({'features': selected_features, 'validation_score': validation_score,
                                 'test_score': test_score})
            rows.append(new_row)
            print(new_row)

        df = pd.concat(rows, axis=1).T
        dump(df, get_save_path(rna_type, file_type='joblib'))
        df.to_csv(get_save_path(rna_type, file_type='csv'))


def xgb_feature_selection(RNAp_X, RNAp_y):
    RUNS = [(RNAp_X, RNAp_y, 'p')]

    for (X, y, rna_type) in RUNS:
        rows = []
        X, y = remove_outliers(X, y)
        seeds = get_seeds(20)
        for seed in tqdm(seeds):
            X_train, X_valid, X_test, y_train, y_valid, y_test = train_validation_test_split(X, y, seed)
            selected_features, test_score = prediction_regressor(X_train, y_train, X_test, y_test)
            model = xgb.XGBRegressor()
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_valid[selected_features])
            validation_score = r2_score(y_valid, y_pred)
            new_row = pd.Series({'features': selected_features, 'validation_score': validation_score,
                                 'test_score': test_score})
            rows.append(new_row)
            print(new_row)

        df = pd.concat(rows, axis=1).T
        dump(df, get_save_path(rna_type, file_type='joblib'))
        df.to_csv(get_save_path(rna_type, file_type='csv'))
        return df



if __name__ == '__main__':
    main()
