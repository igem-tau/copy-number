import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Lasso
from src.consts import *
from src.data_prep.pre_process import get_features_df
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model
from src.models.Parameters_Tuning.best_param_to_xl import get_best_param_optuna
from src.utils import get_current_file_parent_path, estimate_pred

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def run_pipeline(rna_type: str):
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = rna_type[0]
    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train']
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val']
    RNA_y_val = data[f'RNA{rna_type}_y_val']

    RNA_X_test_features = data[f'RNA{rna_type}_X_test']
    RNA_y_test = data[f'RNA{rna_type}_y_test']

    if rna_type == 'i_w_folding':
        low_variance_features = RNA_X_train_features.var().sort_values().iloc[:RNA_X_val_features.shape[1] // 2].index
        RNA_X_train_features = RNA_X_train_features.drop(columns=low_variance_features)
        RNA_X_val_features = RNA_X_val_features.drop(columns=low_variance_features)
        RNA_X_test_features = RNA_X_test_features.drop(columns=low_variance_features)

    # Feature and model selection
    param_dict = model_selection(RNA_X_train_features, RNA_X_val_features, RNA_y_train, RNA_y_val, rna_type)
    models = ['XGBoost', 'CatBoostRegressor', 'LGBMRegressor', 'RandomForest']

    val_predictions = {}
    total_pred = {}
    for cur_model_name in models:
        RNA_selected_features = feature_selection(RNA_X_train_features, RNA_y_train, param_dict, cur_model_name,
                                                  rna_type)

        RNA_FS_train = RNA_X_train_features[RNA_selected_features]
        RNA_FS_val = RNA_X_val_features[RNA_selected_features]
        RNA_FS_test = RNA_X_test_features[RNA_selected_features]

        # Hyperparameters tuning
        Best_params = get_best_param_optuna(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, cur_model_name, rna_type)

        # Run model
        # TODO - Recalculate train_val with selected_features only while using the entire dataset
        RNA_FS_train_val_X = pd.concat([RNA_FS_train, RNA_FS_val])
        RNA_train_val_y = pd.concat([RNA_y_train, RNA_y_val])
        print('++++++++++++++++++++++++++++++++++++++++++++++++')
        trained_model, r2, mae_score, pearson, spearman, y_pred = model(RNA_FS_train_val_X, RNA_FS_test,
                                                                        RNA_train_val_y, RNA_y_test,
                                                                        cur_model_name, f'RNA{rna_type}', Best_params,
                                                                        save_plots=False)
        total_pred[cur_model_name] = y_pred
        print('++++++++++++++++++++++++++++++++++++++++++++++++')

        print('------------------------------------------------')
        _, _, _, _, _, y_val_pred = model(RNA_FS_train, RNA_FS_val,
                                                                        RNA_y_train, RNA_y_val,
                                                                        cur_model_name, f'RNA{rna_type}', Best_params,
                                                                        save_plots=False)
        print('------------------------------------------------')
        val_predictions[cur_model_name] = y_val_pred

    print()
    print()

    voting_regression_model = Lasso()
    voting_regression_model.fit(pd.DataFrame(val_predictions), RNA_y_val)
    voting_results = voting_regression_model.predict(pd.DataFrame(total_pred))
    estimate_pred(RNA_y_test, voting_results, 'regression voting model', data_title=f'RNA{rna_type}', save_plots=True)


if __name__ == '__main__':
    # run_pipeline(rna_type='p')
    # run_pipeline(rna_type='p_fitted')
    # run_pipeline(rna_type='i')
    run_pipeline(rna_type='i_w_folding')
