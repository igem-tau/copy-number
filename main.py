import os
import pandas as pd
from pathlib import Path
from src.analysis.EDA import EDA
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Features_Models_Selection import feature_selection, model_selection
from src.models.models_functions import model, scale
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, get_best_param_optuna
from src.models.sequences_generator import sequence_df_generator
from src.utils import get_current_file_parent_path, write_selected_features

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, 'data')


if __name__ == '__main__':
    # Load the data features if exists, write if it doesn't
    data = get_features_df(p=True, i=False)

    # Extract X, Y, sequences - DataFrames

    RNAp_stratify_train_val = data['RNAp_stratify_train_val']

    RNAp_X_train_sequences = data['RNAp_X_train_sequences']
    RNAp_X_train_features = data['RNAp_X_train']
    RNAp_y_train = data['RNAp_y_train']

    RNAp_X_val_sequences = data['RNAp_X_val_sequences']
    RNAp_X_val_features = data['RNAp_X_val']
    RNAp_y_val = data['RNAp_y_val']

    RNAp_X_test_sequences = data['RNAp_X_test_sequences']
    RNAp_X_test_features = data['RNAp_X_test']
    RNAp_y_test = data['RNAp_y_test']
    RNAp_stratify_test = data['RNAp_stratify_test']

    # Feature and model selection
    param_dict = model_selection(RNAp_X_train_features, RNAp_X_val_features, RNAp_y_train, RNAp_y_val)
    models_fs_data = feature_selection(RNAp_X_train_features, RNAp_y_train, param_dict)

    models = ['XGBoost'] # , 'CatBoostRegressor']
    total_pred = []
    final_predicted_dfs = []
    for cur_model_name in models:
        RNAp_selected_features_data = models_fs_data[cur_model_name]
        RNAp_selected_features = RNAp_selected_features_data['selected_features']

        # write selected features to file
        write_selected_features(RNAp_selected_features)

        # Data by selected features
        RNAp_FS_train = RNAp_X_train_features[RNAp_selected_features]
        RNAp_FS_val = RNAp_X_val_features[RNAp_selected_features]
        RNAp_FS_test = RNAp_X_test_features[RNAp_selected_features]

        # Exploratory Data Analysis (EDA)
        # EDA(RNAp_FS_train, RNAp_FS_val, RNAp_y_train, RNAp_y_val)

        # Hyperparameters tuning
        Best_param_p = get_best_param_optuna(RNAp_FS_train, RNAp_FS_val, RNAp_y_train, RNAp_y_val, cur_model_name)

        # Run model
        # TODO - Recalculate train_val with selected_features only while using the entire dataset
        RNAp_FS_train_val_X = pd.concat([RNAp_FS_train, RNAp_FS_val])
        RNAp_train_val_y = pd.concat([RNAp_y_train, RNAp_y_val])
        trained_model, r2, mae_score, pearson, spearman, y_pred = model(RNAp_FS_train_val_X, RNAp_FS_test, RNAp_train_val_y, RNAp_y_test,
                                       cur_model_name, 'pRNA', Best_param_p, save_plots=False)
        total_pred.append(y_pred)

        # Generate sequences and calculate features
        generated_RNAp_df = sequence_df_generator(rna_type='p')

        # Generate selected features
        USE_SELECTED_FEATURES["selective"] = True
        RNAp_train_val_data = pd.concat(
            (pd.concat((RNAp_X_train_features, RNAp_X_val_features)), pd.concat((RNAp_y_train, RNAp_y_val))),
            axis=1
        )
        # cp is False because RNA_y should be None because we need to predict the copy number
        all_seqs_selected_features, _ = generate_features(generated_RNAp_df, reference_RNA_data=RNAp_train_val_data,
                                                          cp=False,model_name=cur_model_name)

        # Predict
        all_seqs_selected_features = all_seqs_selected_features[RNAp_selected_features]
        _, all_seqs_selected_features_scaled = scale(RNAp_FS_train_val_X, all_seqs_selected_features)
        y_pred = trained_model.predict(all_seqs_selected_features_scaled)
        print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")

        final_predicted_df = generated_RNAp_df[['Promoter Sequence (-35 to +1)']].join(pd.DataFrame({"copy number": y_pred}))
        final_predicted_dfs.append(final_predicted_df)
        final_predicted_df.to_csv(f"copy_num_predictions_{cur_model_name}.csv", index=False)

        # save models
        trained_model.save_model(os.path.join(DATA_PATH, f'{str(pd.to_datetime("today")).split()[0]}_{cur_model_name}_RNAp_model.json'))

    # TODO: combine the two models prediction


