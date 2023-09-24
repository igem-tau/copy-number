import os
import pandas as pd
from pathlib import Path
from src.analysis.EDA import exploratory_data_analysis
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model, scale
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, get_best_param_optuna
from src.models.sequences_generator import sequence_df_generator
from src.utils import get_current_file_parent_path, write_selected_features, get_current_date

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')


def run_pipeline(rna_type: str):
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = rna_type
    data = get_features_df(rna_type=rna_type)

    RNA_X_train_features = data[f'RNA{rna_type}_X_train']
    RNA_y_train = data[f'RNA{rna_type}_y_train']

    RNA_X_val_features = data[f'RNA{rna_type}_X_val']
    RNA_y_val = data[f'RNA{rna_type}_y_val']

    RNA_X_test_features = data[f'RNA{rna_type}_X_test']
    RNA_y_test = data[f'RNA{rna_type}_y_test']

    # Feature and model selection
    param_dict = model_selection(RNA_X_train_features, RNA_X_val_features, RNA_y_train, RNA_y_val, rna_type)
    models = ['XGBoost']  # , 'CatBoostRegressor']
    models_fs_data = feature_selection(RNA_X_train_features, RNA_y_train, param_dict, models, rna_type)

    total_pred = []
    final_predicted_dfs = []
    for cur_model_name in models:
        RNA_selected_features_data = models_fs_data[cur_model_name]
        RNA_selected_features = RNA_selected_features_data['selected_features']

        # write selected features to file
        write_selected_features(RNA_selected_features, rna_type)  # TODO - input the selected model

        # Data by selected features
        RNA_FS_train = RNA_X_train_features[RNA_selected_features]
        RNA_FS_val = RNA_X_val_features[RNA_selected_features]
        RNA_FS_test = RNA_X_test_features[RNA_selected_features]

        # Exploratory Data Analysis (EDA)
        if rna_type == 'p':
            exploratory_data_analysis(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, rna_type)

        # Hyperparameters tuning
        # Best_param_xgb = get_best_params_set_xgb(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val,
        #                                          f'xgb_RNA{rna_type}')
        Best_params = get_best_param_optuna(RNA_FS_train, RNA_FS_val, RNA_y_train, RNA_y_val, cur_model_name)

        # Run model
        # TODO - Recalculate train_val with selected_features only while using the entire dataset
        RNA_FS_train_val_X = pd.concat([RNA_FS_train, RNA_FS_val])
        RNA_train_val_y = pd.concat([RNA_y_train, RNA_y_val])
        trained_model, r2, mae_score, pearson, spearman, y_pred = model(RNA_FS_train_val_X, RNA_FS_test,
                                                                        RNA_train_val_y, RNA_y_test,
                                                                        cur_model_name, f'RNA{rna_type}', Best_params,
                                                                        save_plots=False)
        total_pred.append(y_pred)

        # Generate sequences and calculate features
        generated_RNA_df = sequence_df_generator(rna_type=rna_type)

        # Generate selected features
        USE_SELECTED_FEATURES["selective"] = True
        RNA_train_val_data = pd.concat(
            (pd.concat((RNA_X_train_features, RNA_X_val_features)), pd.concat((RNA_y_train, RNA_y_val))),
            axis=1
        )
        # cp is False because RNA_y should be None because we need to predict the copy number
        all_seqs_selected_features, _ = generate_features(generated_RNA_df, rna_type=rna_type,
                                                          reference_RNA_data=RNA_train_val_data, cp=False)

        # Predict
        all_seqs_selected_features = all_seqs_selected_features[RNA_selected_features]
        _, all_seqs_selected_features_scaled = scale(RNA_FS_train_val_X, all_seqs_selected_features)
        y_pred = trained_model.predict(all_seqs_selected_features_scaled)
        print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")

        final_predicted_df = generated_RNA_df[['Promoter Sequence (-35 to +1)']].join(
            pd.DataFrame({'copy number': y_pred}))
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


if __name__ == '__main__':
    # run_pipeline(rna_type='p')
    run_pipeline(rna_type='i')
