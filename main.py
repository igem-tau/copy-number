from pathlib import Path
from src.utils import get_current_file_parent_path
from src.data_prep.pre_process import get_RNAp_data, split_for_testing, create_fasta_file, get_features_df
from src.models.Feature_Selection import feature_selection
from src.models.sequences_generator import sequence_generator
from xgboost import XGBRegressor
from src.consts import *
from src.data_prep.pre_process import get_features_df, generate_features
from src.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, find_optimal_alpha_Lasso
from src.models.models_functions import model


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


# Create the FASTA file


if __name__ == '__main__':
    # Load the data features

    data = get_features_df(p=True, i=False)

    RNAp_X_train_val_data = data['RNAp_X_train_val_sequences']
    RNAp_X_train_val_features = data['RNAp_X_train_val']
    RNAp_y_train_val = data['RNAp_y_train_val']
    RNAp_stratify_train_val = data['RNAp_stratify_train_val']

    RNAp_X_train_sequences = data['RNAp_X_train_sequences']
    RNAp_X_train_features = data['RNAp_X_train']
    RNAp_y_train = data['RNAp_y_train']
    RNAp_stratify_train = data['RNAp_stratify_train']

    RNAp_X_val_sequences = data['RNAp_X_val_sequences']
    RNAp_X_val_features = data['RNAp_X_val']
    RNAp_y_val = data['RNAp_y_val']
    RNAp_stratify_val = data['RNAp_stratify_val']

    RNAp_X_test_sequences = data['RNAp_X_test_sequences']
    RNAp_X_test_features = data['RNAp_X_test']
    RNAp_y_test = data['RNAp_y_test']
    RNAp_stratify_test = data['RNAp_stratify_test']

    # Feature selection
    RNAp_FS_train, RNAp_selected_features, RNAp_removed_features = feature_selection(RNAp_X_train_features, RNAp_y_train)

    # Data by selected features
    RNAp_FS_train_val = RNAp_X_train_val_features[RNAp_selected_features]
    RNAp_FS_val = RNAp_X_val_features[RNAp_selected_features]
    RNAp_FS_test = RNAp_X_test_features[RNAp_selected_features]

    # Hyperparameters tuning and model training
    Best_param_p_xgb = get_best_params_set_xgb(RNAp_FS_train_val, RNAp_y_train_val, 'xgb_RNAp', RNAp_stratify_train_val)

    # Run model
    model(RNAp_FS_train_val, RNAp_FS_test, RNAp_y_train_val, RNAp_y_test, 'xgboost', 'pRNA', Best_param_p_xgb, save_plots=True)

    # Generate sequences and calculate features
    generated_RNAp_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type='p')

    USE_SELECTED_FEATURES["selective"] = True
    all_seqs_selected_features = generate_features(generated_RNAp_seq)

    # Predict








