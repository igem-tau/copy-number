import pandas as pd
from pathlib import Path
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Feature_Selection import feature_selection
from src.models.models_functions import model
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb
from src.models.sequences_generator import sequence_df_generator
from src.utils import get_current_file_parent_path, write_selected_features

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


# Create the FASTA file


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

    # Feature selection
    RNAp_selected_features_data = feature_selection(RNAp_X_train_features, RNAp_y_train)
    RNAp_selected_features = RNAp_selected_features_data['selected_features']

    # write selected features to file
    write_selected_features(RNAp_selected_features)

    # Data by selected features
    RNAp_FS_train = RNAp_X_train_features[RNAp_selected_features]
    RNAp_FS_val = RNAp_X_val_features[RNAp_selected_features]
    RNAp_FS_test = RNAp_X_test_features[RNAp_selected_features]

    # Hyperparameters tuning
    Best_param_p_xgb = get_best_params_set_xgb(RNAp_FS_train, RNAp_FS_val, RNAp_y_train, RNAp_y_val, 'xgb_RNAp', RNAp_stratify_train_val)

    # Run model
    # TODO - Recalculate train_val with selected_features only while using the entire dataset
    trained_model, _, _, _ = model(pd.concat([RNAp_FS_train, RNAp_FS_val]), RNAp_FS_test, pd.concat([RNAp_y_train, RNAp_y_val]), RNAp_y_test,
                                   'xgboost', 'pRNA', Best_param_p_xgb, save_plots=True)

    # Generate sequences and calculate features
    generated_RNAp_df = sequence_df_generator(rna_type='p')

    generated_RNAp_df = generated_RNAp_df.head(100)

    # Generate selected features
    USE_SELECTED_FEATURES["selective"] = True
    all_seqs_selected_features, RNA_y = generate_features(generated_RNAp_df, cp=False)  # cp is False because RNA_y should be None because we need to predict the copy number

    # print(all_seqs_selected_features.columns)
    # print(f"num columns: {len(all_seqs_selected_features.columns)}")
    # print(f"columns == selected features: {set(RNAp_selected_features) == set(all_seqs_selected_features.columns)}")

    # Predict
    all_seqs_selected_features = all_seqs_selected_features[RNAp_selected_features]
    y_pred = trained_model.predict(all_seqs_selected_features)
    print(f"Range of copy nums predicted: {y_pred.min()} - {y_pred.max()}")

    final_predicted_df = generated_RNAp_df[['Promoter Sequence (-35 to +1)']].join(pd.DataFrame({"copy number": y_pred}))
    final_predicted_df.to_csv("copy_num_predictions.csv")












