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

def run_RNAp():
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = 'p'
    data = get_features_df(rna_type = 'p')

    # Extract X, Y, sequences - DataFrames

    # RNAp

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



def run_RNAi():
    # Load the data features if exists, write if it doesn't
    RNA_TYPE_CONST['RNA'] = 'i'
    data = get_features_df(rna_type = 'i')

    # Extract X, Y, sequences - DataFrames

    RNAi_stratify_train_val = data['RNAi_stratify_train_val']

    RNAi_X_train_sequences = data['RNAi_X_train_sequences']
    RNAi_X_train_features = data['RNAi_X_train']
    RNAi_y_train = data['RNAi_y_train']

    RNAi_X_val_sequences = data['RNAi_X_val_sequences']
    RNAi_X_val_features = data['RNAi_X_val']
    RNAi_y_val = data['RNAi_y_val']

    RNAi_X_test_sequences = data['RNAi_X_test_sequences']
    RNAi_X_test_features = data['RNAi_X_test']
    RNAi_y_test = data['RNAi_y_test']
    RNAi_stratify_test = data['RNAi_stratify_test']

    # Feature selection
    RNAi_selected_features_data = feature_selection(RNAi_X_train_features, RNAi_y_train, 'i')
    RNAi_selected_features = RNAi_selected_features_data['selected_features']

    # write selected features to file
    write_selected_features(RNAi_selected_features, 'i')

    # Data by selected features
    RNAi_FS_train = RNAi_X_train_features[RNAi_selected_features]
    RNAi_FS_val = RNAi_X_val_features[RNAi_selected_features]
    RNAi_FS_test = RNAi_X_test_features[RNAi_selected_features]

    # Exploratory Data Analysis (EDA)
    EDA(RNAi_FS_train, RNAi_FS_val, RNAi_y_train, RNAi_y_val)

    # Hyperparameters tuning
    # Best_param_p_xgb = get_best_params_set_xgb(RNAi_FS_train, RNAi_FS_val, RNAi_y_train, RNAi_y_val, 'xgb_RNAi',
    #                                            RNAi_stratify_train_val)
    Best_param_p_xgb = get_best_param_optuna(RNAi_FS_train, RNAi_FS_val, RNAi_y_train, RNAi_y_val)


    # Run model
    # TODO - Recalculate train_val with selected_features only while using the entire dataset
    RNAi_FS_train_val_X = pd.concat([RNAi_FS_train, RNAi_FS_val])
    RNAi_train_val_y = pd.concat([RNAi_y_train, RNAi_y_val])
    trained_model, _, _, _ = model(RNAi_FS_train_val_X, RNAi_FS_test, RNAi_train_val_y, RNAi_y_test,
                                   'xgboost', 'iRNA', Best_param_p_xgb, save_plots=True)

    # Generate sequences and calculate features
    generated_RNAi_df = sequence_df_generator(rna_type='i')

    # Generate selected features
    USE_SELECTED_FEATURES["selective"] = True
    RNAi_train_val_data = pd.concat(
        (pd.concat((RNAi_X_train_features, RNAi_X_val_features)), pd.concat((RNAi_y_train, RNAi_y_val))),
        axis=1
    )
    # cp is False because RNA_y should be None because we need to predict the copy number
    all_seqs_selected_features, _ = generate_features(generated_RNAi_df, reference_RNA_data=RNAi_train_val_data,
                                                      cp=False)

    # Predict
    all_seqs_selected_features = all_seqs_selected_features[RNAi_selected_features]
    _, all_seqs_selected_features_scaled = scale(RNAi_FS_train_val_X, all_seqs_selected_features)
    y_pred = trained_model.predict(all_seqs_selected_features_scaled)
    print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")

    final_predicted_df = generated_RNAi_df[['Promoter Sequence (-35 to +1)']].join(
        pd.DataFrame({"copy number": y_pred}))
    final_predicted_df.to_csv("RNAi_copy_num_predictions.csv", index=False)



if __name__ == '__main__':
    run_RNAp()
    # run_RNAi()

