import os
import re

import pandas as pd
from pathlib import Path
from src.analysis.EDA import exploratory_data_analysis
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model, scale, estimate_pred
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, get_best_param_optuna
from src.models.sequences_generator import sequence_df_generator
from src.utils import get_current_file_parent_path, get_current_date
import numpy as np

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')


def run_pipeline(rna_type: str):
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = rna_type
    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train']
    RNA_X_train_seq = data[f'RNA{rna_type}_X_train_sequences']
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val']
    RNA_X_val_seq = data[f'RNA{rna_type}_X_val_sequences']
    RNA_y_val = data[f'RNA{rna_type}_y_val']

    RNA_X_test_features = data[f'RNA{rna_type}_X_test']
    RNA_y_test = data[f'RNA{rna_type}_y_test']


    # Feature and model selection
    param_dict = model_selection(RNA_X_train_features, RNA_X_val_features, RNA_y_train, RNA_y_val, rna_type)
    models = ['XGBoost', 'CatBoostRegressor',  'RandomForest']

    total_pred = []
    final_predicted_dfs = []
    for cur_model_name in models:
        RNA_selected_features_data = feature_selection(RNA_X_train_features, RNA_y_train, param_dict, cur_model_name, rna_type)
        RNA_selected_features = RNA_selected_features_data['selected_features']

        # Data by selected features
        if cur_model_name=='LGBMRegressor':
            RNA_X_train_features.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            RNA_X_val_features.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            RNA_X_test_features.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        RNA_FS_train = RNA_X_train_features[RNA_selected_features]
        RNA_FS_val = RNA_X_val_features[RNA_selected_features]
        RNA_FS_test = RNA_X_test_features[RNA_selected_features]

        # Hyperparameters tuning
        Best_params = get_best_param_optuna(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, cur_model_name, rna_type)

        # Run model
        # TODO - Recalculate train_val with selected_features only while using the entire dataset
        RNA_FS_train_val_X = pd.concat([RNA_FS_train, RNA_FS_val])
        RNA_train_val_seq = pd.concat([RNA_X_train_seq, RNA_X_val_seq])
        RNA_train_val_y = pd.concat([RNA_y_train, RNA_y_val])
        trained_model, r2, mae_score, pearson, spearman, y_pred = model(RNA_FS_train_val_X, RNA_FS_test,
                                                                        RNA_train_val_y, RNA_y_test,
                                                                        cur_model_name, f'RNA{rna_type}', Best_params,
                                                                        save_plots=True)
        total_pred.append(y_pred)

        # Exploratory Data Analysis (EDA)
        # exploratory_data_analysis(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, rna_type)

        # Generate selected features
        RNA_train_val_data = pd.concat(
            (pd.concat((RNA_X_train_features, RNA_X_val_features)), pd.concat((RNA_y_train, RNA_y_val))),
            axis=1
        )
        # cp is False because RNA_y should be None because we need to predict the copy number
        RNA_train_val_data_seq = pd.concat([RNA_train_val_data.reset_index(drop=True), RNA_train_val_seq['Promoter Sequence (-35 to +1)'].reset_index(drop=True)], axis=1)
        all_seqs_selected_features, _ = generate_features(generated_RNA_df, rna_type=rna_type,
                                                          reference_RNA_data=RNA_train_val_data_seq, cp=False,
                                                          selected_features=RNA_selected_features)

        # Predict
        _, all_seqs_selected_features_scaled = scale(RNA_FS_train_val_X, all_seqs_selected_features)
        y_pred = trained_model.predict(all_seqs_selected_features_scaled)
        print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")

        if TARGET_COLUMN == 'Raw Copy Number':
            final_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
                pd.DataFrame({'Copy Number': np.exp(y_pred)}))
        else:
            final_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
                pd.DataFrame({'Copy Number': y_pred}))
        final_predicted_dfs.append(final_predicted_df)
        final_predicted_df.to_csv(f'copy_num_predictions_RNA{rna_type}_{cur_model_name}.csv', index=False)

        # save models
        trained_model.save_model(
            os.path.join(
                DATA_PATH,
                f'{get_current_date()}_{cur_model_name}_RNA{rna_type}_model.json'
            )
        )

    # TODO: combine the two models prediction
    final_pred = np.array(total_pred).mean(axis=0)
    estimate_pred(RNA_y_test, final_pred, 'voting model', rna_type = rna_type)

    # final_predicted_df


if __name__ == '__main__':
    run_pipeline(rna_type='p')
    # run_pipeline(rna_type='i')
    # run_pipeline(rna_type='i_w_folding')
