from pathlib import Path
from src.analysis.forward_features_selection import multi_split_forward_selection
from src.utils import get_current_file_parent_path
from src.data_prep.pre_process import get_RNAp_data, split_for_testing, create_fasta_file, get_features_df
from xgboost import XGBRegressor
from src.models.sequences_generator import sequence_generator

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


# Create the FASTA file


if __name__ == '__main__':
    # load data with features
    data = get_features_df(p=True, i=False)
    RNAp_X_train_val = data['RNAp_X_train_val']
    RNAp_y_train_val = data['RNAp_y_train_val']
    RNAp_X_test = data['RNAp_X_test']
    RNAp_y_test = data['RNAp_y_test']

    # features selection
    fs_df = multi_split_forward_selection(data, rna_type='p', model=XGBRegressor(), model_type='xgboost')
    chosen_features = []
    # # hyperparameters tuning

    # train model

    # validation

    # save model and list of features

    # Generate sequences and calculate features
    generated_RNAp_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type='p')
    generated_RNAi_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type = 'i')

