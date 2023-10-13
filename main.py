import pandas as pd
from pathlib import Path
from joblib import load, dump
from src.analysis.EDA import exploratory_data_analysis
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model, scale
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, get_best_param_optuna
from src.models.sequences_generator import sequence_df_generator
from src.utils import get_current_file_parent_path, get_current_date, estimate_pred
import numpy as np

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def run_pipeline(rna_type: str):
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = rna_type[0]
    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train']
    RNA_X_train_seq = data[f'RNA{rna_type}_X_train_sequences']
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val']
    RNA_X_val_seq = data[f'RNA{rna_type}_X_val_sequences']
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

    if rna_type[0] == 'p':
        # Generate sequences and calculate features
        generated_RNA_df = sequence_df_generator(rna_type=rna_type)

    total_pred = []
    final_predicted_dfs = []
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
        RNA_train_val_seq = pd.concat([RNA_X_train_seq, RNA_X_val_seq])
        RNA_train_val_y = pd.concat([RNA_y_train, RNA_y_val])
        trained_model, r2, mae_score, pearson, spearman, y_pred = model(RNA_FS_train_val_X, RNA_FS_test,
                                                                        RNA_train_val_y, RNA_y_test,
                                                                        cur_model_name, f'RNA{rna_type}', Best_params,
                                                                        save_plots=True)
        total_pred.append(y_pred)

        # Exploratory Data Analysis (EDA)
        exploratory_data_analysis(cur_model_name, RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, rna_type)

        if rna_type[0] == 'p':
            # Generate selected features
            all_seqs_selected_features = pd.DataFrame()
            all_seqs_features_file_path = Path(DATA_PATH, f'RNA{rna_type[0]}_all_sequences_features.joblib')
            if all_seqs_features_file_path.exists():
                all_seqs_selected_features = load(all_seqs_features_file_path)

            selected_features_left_to_generate = set(RNA_selected_features)
            selected_features_left_to_generate.difference_update(set(all_seqs_selected_features.columns))

            if len(selected_features_left_to_generate) > 0:
                RNA_train_val_data = pd.concat(
                    (pd.concat((RNA_X_train_features, RNA_X_val_features)), pd.concat((RNA_y_train, RNA_y_val))),
                    axis=1
                )
                # cp is False because RNA_y should be None because we need to predict the copy number
                RNA_train_val_data_seq = pd.concat([RNA_train_val_data.reset_index(drop=True),
                                                    RNA_train_val_seq['Promoter Sequence (-35 to +1)'].reset_index(
                                                        drop=True)],
                                                   axis=1)
                all_seqs_additional_selected_features, _ = generate_features(generated_RNA_df, rna_type=rna_type,
                                                                             reference_RNA_data=RNA_train_val_data_seq,
                                                                             cp=False,
                                                                             selected_features=selected_features_left_to_generate)
                all_seqs_selected_features = pd.concat(
                    (all_seqs_selected_features, all_seqs_additional_selected_features),
                    axis=1)
                dump(all_seqs_selected_features, all_seqs_features_file_path, compress=True)

            all_seqs_selected_features = all_seqs_selected_features[RNA_selected_features]

            # Predict
            _, all_seqs_selected_features_scaled = scale(RNA_FS_train_val_X, all_seqs_selected_features)
            y_pred = trained_model.predict(all_seqs_selected_features_scaled)

            # zero negative copy number predictions
            y_pred[y_pred < 0] = 0
            print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")
            print()

            if TARGET_COLUMN == 'Raw Copy Number':
                final_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
                    pd.DataFrame({'Copy Number': y_pred if rna_type == 'p_fitted' else np.exp(y_pred)}))
            elif TARGET_COLUMN == 'Copy Number':
                final_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
                    pd.DataFrame({'Copy Number': y_pred}))
            else:
                raise ValueError(
                    'main: TARGET_COLUMN must be one of the following values: "Copy Number" or "Raw Copy Number"')

            final_predicted_df.to_csv(f'copy_num_predictions_RNA{rna_type}_{cur_model_name}.csv', index=False)
            final_predicted_dfs.append(final_predicted_df)

        # save models
        model_file_name = f'{get_current_date()}_{cur_model_name}_RNA{rna_type}_model'
        try:
            model_file_name += '.json'
            model_path = Path(DATA_PATH, model_file_name)
            trained_model.save_model(model_path)
        except AttributeError:
            model_file_name += '.joblib'
            model_path = Path(DATA_PATH, model_file_name)
            dump(trained_model, model_path, compress=True)

    # TODO: combine the two models prediction
    final_pred = np.array(total_pred).mean(axis=0)
    estimate_pred(RNA_y_test, final_pred, 'voting model', data_title=f'RNA{rna_type}')

    if rna_type[0] == 'p':
        # final_predicted_df
        all_seq_voting_pred = pd.concat([df['Copy Number'] for df in final_predicted_dfs], axis=1).mean(axis=1)
        final_voting_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
            pd.DataFrame({'Copy Number': all_seq_voting_pred}))
        final_voting_predicted_df.to_csv(f'copy_num_predictions_RNA{rna_type}_voting.csv', index=False)


if __name__ == '__main__':
    # run_pipeline(rna_type='p')
    # run_pipeline(rna_type='p_fitted')
    # run_pipeline(rna_type='i')
    run_pipeline(rna_type='i_w_folding')
