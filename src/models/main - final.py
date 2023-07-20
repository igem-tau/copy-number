from joblib import dump
import os
import pandas as pd

from src.analysis.feature_selection_with_XGB import xgb_feature_selection
from src.data_prep.pre_process import get_RNAp_data, split_for_validation, \
    generate_features, remove_zero_variance_features, create_fasta_file

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, '..',  '..', 'data')





# Create the FASTA file


if __name__ == '__main__':
    # load data
    RNAp_data = get_RNAp_data()

    # split data to train and validation
    RNAp_data, RNAp_data_val = split_for_validation(RNAp_data)
    create_fasta_file(RNAp_data)

    # features extraction
    data = {}
    RNAp_X, RNAp_y = generate_features(RNAp_data, type='p', val=False)
    data['RNAp_X'] = remove_zero_variance_features(RNAp_X)
    data['RNAp_y'] = RNAp_y
    dump(data, os.path.join(DATA_PATH, 'DataFrames_with_features.joblib'))
    pd.concat([data['RNAp_X'], data['RNAp_y']]).to_csv(os.path.join(DATA_PATH, 'p_RNA_DataFrames_with_features.csv'))

    # features selection
    fs_df = xgb_feature_selection(RNAp_X, RNAp_y)
    chosen_features = []
    # hyperparameters tuning

    # train model

    # validation

    # save model and list of features