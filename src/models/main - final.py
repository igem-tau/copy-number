from pathlib import Path
from src.analysis.forward_features_selection import multi_split_forward_selection
from src.utils import get_current_file_parent_path
from src.data_prep.pre_process import get_RNAp_data, split_for_testing, create_fasta_file, get_features_df
from src.models.Feature_Selection import feature_selection
from src.models.sequences_generator import sequence_generator
from xgboost import XGBRegressor


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


# Create the FASTA file


if __name__ == '__main__':
    # Load the data features
    data = get_features_df(p=True, i=True)

    RNAp_X_train_val = data['RNAp_X_train_val']
    RNAp_y_train_val = data['RNAp_y_train_val']
    RNAp_X_test = data['RNAp_X_test']
    RNAp_y_test = data['RNAp_y_test']

    RNAi_X_train_val = data['RNAi_X_train_val']
    RNAi_y_train_val = data['RNAi_y_train_val']

    RNAi_X_test = data['RNAi_X_test']
    RNAi_y_test = data['RNAi_y_test']

    # Feature selection
    RNAp_after_FS, RNAp_selected_features, RNAp_removed_features = feature_selection(RNAp_X_train_val, RNAp_y_train_val)
    RNAi_after_FS, RNAi_selected_features, RNAi_removed_features = feature_selection(RNAi_X_train_val, RNAi_y_train_val)

    # Data by selected features

    # Hyperparameters tuning and model training

    # Train model

    # Validation

    # Test



    # Generate sequences and calculate features
    generated_RNAp_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type='p')
    generated_RNAi_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type='i')

    # Predict








