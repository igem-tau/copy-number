import numpy as np
import pandas as pd
from pathlib import Path

from joblib import dump, load
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from src.consts import TARGET_COLUMN, RNA_TYPE_CONST
from src.data_prep.pre_process import split_into_percentages, get_RNAp_data, equal_bins_data, \
    generate_features, remove_zero_variance_features
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model
from src.models.Parameters_Tuning.best_param_to_xl import get_best_param_optuna
from src.utils import get_current_file_parent_path, estimate_pred

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')
OUTPUT_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'output')
FEATURES_DATA_PATH = Path(OUTPUT_PATH, 'data_with_features.joblib')
MODELS_PATH = Path(OUTPUT_PATH, 'models.joblib')

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def run_pipeline(rna_type: str):
    if not FEATURES_DATA_PATH.exists():
        # Load the data features if exists, write if it doesn't
        RNA_TYPE_CONST['RNA'] = rna_type[0]
        RNA_data = get_RNAp_data(rna_type[0])
        RNA_data, RNA_stratify_col = equal_bins_data(RNA_data)
        X, y = RNA_data.drop(TARGET_COLUMN, axis=1), RNA_data[TARGET_COLUMN]
        [(X_train, y_train), (X_val1, y_val1), (X_val2, y_val2), (X_test, y_test)] = split_into_percentages(
            X, y, stratify_by=RNA_stratify_col, percentages=[0.7, 0.1, 0.1, 0.1]
        )
        train_data = pd.concat([X_train, pd.DataFrame(y_train, columns=[TARGET_COLUMN])], axis=1)
        X_train, y_train = generate_features(train_data, rna_type=rna_type)
        X_train = remove_zero_variance_features(X_train)
        X_train_features = X_train.columns.values

        val1_data = pd.concat([X_val1, pd.DataFrame(y_val1, columns=[TARGET_COLUMN])], axis=1)
        X_val1, y_val1 = generate_features(val1_data, rna_type=rna_type, reference_RNA_data=train_data,
                                           selected_features=X_train_features)
        X_val1 = X_val1[X_train_features]
        X_train_val1 = pd.concat([X_train, X_val1], ignore_index=True)
        y_train_val1 = pd.concat([y_train, y_val1], ignore_index=True)

        train_val1_data = pd.concat([train_data, val1_data], ignore_index=True)
        val2_data = pd.concat([X_val2, pd.DataFrame(y_val2, columns=[TARGET_COLUMN])], axis=1)
        X_val2, y_val2 = generate_features(val2_data, rna_type=rna_type, reference_RNA_data=train_val1_data,
                                           selected_features=X_train_features)
        X_val2 = X_val2[X_train_features]
        X_train_val1_val2 = pd.concat([X_train_val1, X_val2], ignore_index=True)
        y_train_val1_val2 = pd.concat([y_train_val1, y_val2], ignore_index=True)

        train_val1_val2_data = pd.concat([train_val1_data, val2_data], ignore_index=True)
        test_data = pd.concat([X_test, pd.DataFrame(y_test, columns=[TARGET_COLUMN])], axis=1)
        X_test, y_test = generate_features(test_data, rna_type=rna_type, reference_RNA_data=train_val1_val2_data,
                                           selected_features=X_train_features)
        X_test = X_test[X_train_features]

        dump(
            (X_train, y_train, X_val1, y_val1, X_train_val1, y_train_val1, X_val2, y_val2, X_train_val1_val2,
             y_train_val1_val2, X_test, y_test),
            FEATURES_DATA_PATH, compress=True
        )
    else:
        X_train, y_train, X_val1, y_val1, X_train_val1, y_train_val1, X_val2, y_val2, X_train_val1_val2, y_train_val1_val2, X_test, y_test = load(
            FEATURES_DATA_PATH)

    # Feature and model selection
    param_dict = model_selection(X_train, X_val1, y_train, y_val1, rna_type)
    models = ['XGBoost', 'CatBoostRegressor', 'LGBMRegressor', 'RandomForest']

    val2_predictions = {}
    test_predictions = {}
    models_tracker = {}
    voting_estimators = []
    voting_features = []
    for cur_model_name in models:
        selected_features = feature_selection(X_train, y_train, param_dict, cur_model_name,
                                              rna_type)

        # Hyperparameters tuning
        Best_params = get_best_param_optuna(X_train[selected_features], X_val1[selected_features], y_train, y_val1,
                                            cur_model_name, rna_type)

        # Run model
        temp_model, _, _, _, _, y_val2_pred = model(X_train_val1[selected_features], X_val2[selected_features],
                                                    y_train_val1, y_val2, cur_model_name, f'RNA{rna_type}', Best_params,
                                                    save_plots=False)
        val2_predictions[cur_model_name] = y_val2_pred

        trained_model, r2, mae_score, pearson, spearman, y_test_pred = model(X_train_val1_val2[selected_features],
                                                                             X_test[selected_features],
                                                                             y_train_val1_val2, y_test, cur_model_name,
                                                                             f'RNA{rna_type}', Best_params,
                                                                             save_plots=False)
        test_predictions[cur_model_name] = y_test_pred

        models_tracker[f'{cur_model_name}:val2'] = temp_model
        models_tracker[f'{cur_model_name}:test'] = trained_model
        voting_estimators.append((cur_model_name, trained_model))
        voting_features.extend(selected_features)
        dump(models_tracker, MODELS_PATH, compress=True)
        print(f'done with: {cur_model_name}')

    # Voting (regression problem of choosing the weights for each model)
    voting_regression_model = Lasso()
    voting_regression_model_2 = VotingRegressor(voting_estimators)
    voting_regression_model.fit(pd.DataFrame(val2_predictions), y_val2)
    voting_features = np.unique(voting_features)
    voting_regression_model_2.fit(X_train_val1_val2[voting_features], y_train_val1_val2)

    voting_results = voting_regression_model.predict(pd.DataFrame(test_predictions))
    voting_results_2 = voting_regression_model_2.predict(X_test[voting_features])
    estimate_pred(y_test, voting_results, 'regression voting model', data_title=f'RNA{rna_type}', save_plots=True)
    estimate_pred(y_test, voting_results_2, 'regression voting model 2', data_title=f'RNA{rna_type}', save_plots=True)

    models_tracker['voting_model'] = voting_regression_model
    models_tracker['voting_model 2'] = voting_regression_model_2
    dump(models_tracker, MODELS_PATH, compress=True)


if __name__ == '__main__':
    run_pipeline(rna_type='p')
