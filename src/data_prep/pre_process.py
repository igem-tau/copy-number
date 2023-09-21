import datetime
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.consts import *
from src.features.denovo_motifs import score_denovo_motifs
# from src.features.motifs import calc_motifs_pv
from src.features.nucleotide_features import generate_one_hot_encoding, entropy, extract_nucli_features
from src.features.promotor_strength import calc_promoter_zones_strength, calc_predicted_promoter_strength
from src.features.pssm_feature import calc_series_pssm_score
from src.features.delta_G.TX_prediction import calculate_dG_and_Tx
from src.utils import get_current_file_parent_path, get_selected_features
import sys
from typing import Optional, Tuple, Union

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
timepoints_df = pd.read_excel(
    Path(DATA_PATH, 'sup_data_3_seq_cnt_p_rna.xlsx'))  # priming RNA time points
PSSM_THRESHOLD_PATH_p = Path(DATA_PATH, 'pssm_threshold_pRNA.pkl')
PSSM_THRESHOLD_PATH_i = Path(DATA_PATH, 'pssm_threshold_iRNA.pkl')
TARGET_COLUMN = 'Copy Number'
RNAp_high_filename = 'pRNA high copy number.fasta'
RNAp_low_filename = 'pRNA low copy number.fasta'
RNAi_high_filename = 'iRNA high copy number.fasta'
RNAi_low_filename = 'iRNA high copy number.fasta'


def get_RNAp_data():
    """
    get RNA_P df, with additional columns
    :return:
    """
    RNAp_df = pd.read_excel(Path(DATA_PATH, 'sup_data_1_p_rna.xlsx'),
                            names=RNA_DATA_COLUMNS)  # priming RNA
    RNAp_df['cnt_grw'] = RNAp_df['Final Counts'] / RNAp_df['Initial Counts']
    shift = abs(RNAp_df[TARGET_COLUMN].min()) + 1e-10
    RNAp_df['copy_num_log'] = np.log(RNAp_df[TARGET_COLUMN] + shift)
    return RNAp_df


def get_RNAi_data():
    """
    get RNA_I df, with additional columns
    :return:
    """
    RNAi_df = pd.read_excel(Path(DATA_PATH, 'sup_data_2_i_rna.xlsx'),
                            names=RNA_DATA_COLUMNS)  # inhibitory RNA
    RNAi_df['cnt_grw'] = RNAi_df['Final Counts'] / RNAi_df['Initial Counts']
    return RNAi_df


def get_RNAi_prom_RNAp():
    df = pd.read_csv(Path(DATA_PATH, 'rna_p_data.csv'), names=RNAi_PROM_RNAp_COLUMNS)
    return df


def get_RNAp_merged_data():
    """
    get RNA_p df merged with the timepoints and additional columns
    :return:
    """
    RNAp_df = pd.read_excel(Path(DATA_PATH, 'sup_data_1_p_rna.xlsx'))  # priming RNA
    timepoints_df = pd.read_excel(
        Path(DATA_PATH, 'sup_data_3_seq_cnt_p_rna.xlsx'))  # priming RNA time points
    RNAp_df['cnt_grw'] = RNAp_df['Final Counts'] / RNAp_df['Initial Counts']
    timepoints_df['avg_dup_rate'] = ((timepoints_df['Timepoint 2 Counts'] / timepoints_df['Timepoint 1 Counts']) +
                                     (timepoints_df['Timepoint 3 Counts'] / timepoints_df['Timepoint 2 Counts']) +
                                     (timepoints_df['Timepoint 4 Counts'] / timepoints_df['Timepoint 3 Counts'])) / 3
    timepoints_df['log_avg_dup_rate'] = np.log(timepoints_df['avg_dup_rate'])
    timepoints_df['s2e_dup_rate'] = (timepoints_df['Timepoint 4 Counts'] / timepoints_df['Timepoint 1 Counts'])
    timepoints_df.rename(columns={'Promoter Sequence (-35 to +1)': 'Promoter Sequence'}, inplace=True)
    merged = pd.merge(RNAp_df, timepoints_df, on='Promoter Sequence')
    return merged


def generate_selected_features(RNA_data: pd.DataFrame, rna_type: str = 'p',
                      reference_RNA_data: Optional[pd.DataFrame] = None, cp: bool = True) -> pd.DataFrame:
    RNA_seq = RNA_data['Promoter Sequence (-35 to +1)']
    RNA_features = []

    selected_features = get_selected_features(RNA_TYPE_CONST['RNA'])
    if 'Predicted Promoter Strength (KbT)' in selected_features:
        RNA_features.append(RNA_data['Predicted Promoter Strength (KbT)'])

    if "pssm_score" in selected_features:
        if reference_RNA_data is not None:
            RNA_pssm_score = calc_series_pssm_score(RNA_data, reference_RNA_data)
        else:
            RNA_pssm_score = calc_series_pssm_score(RNA_data, RNA_data)
        RNA_features.append(RNA_pssm_score)

    # RNA_features.append(calc_motifs_pv(RNA_seq))
    RNA_features.append(generate_one_hot_encoding(RNA_seq))
    RNA_features.append(extract_nucli_features(RNA_seq))
    RNA_features.append(calc_promoter_zones_strength(RNA_seq, RNAp_EDITED_ZONES if rna_type == 'p' else RNAi_EDITED_ZONES))
    RNA_features.append(score_denovo_motifs(RNA_seq))
    RNA_features.append(calculate_dG_and_Tx(RNA_seq))  # 3 features based on biophysical properties (deltaG)

    if "entropy" in selected_features:
        RNA_features.append(entropy(RNA_seq))

    RNA_X = pd.concat(RNA_features, axis=1)
    RNA_X.replace(-np.inf, -sys.maxsize, inplace=True)
    RNA_y = RNA_data[TARGET_COLUMN] if cp else None
    return RNA_X, RNA_y


def generate_features(RNA_data: pd.DataFrame, rna_type: str = 'p',
                      reference_RNA_data: Optional[pd.DataFrame] = None, cp: bool = True) -> pd.DataFrame:
    if USE_SELECTED_FEATURES["selective"]:
        return generate_selected_features(RNA_data, rna_type, reference_RNA_data, cp)

    RNA_seq = RNA_data['Promoter Sequence (-35 to +1)']
    RNA_features = []

    RNA_features.append(RNA_data['Predicted Promoter Strength (KbT)'])

    if reference_RNA_data is not None:
        RNA_pssm_score = calc_series_pssm_score(RNA_data, reference_RNA_data)
    else:
        RNA_pssm_score = calc_series_pssm_score(RNA_data, RNA_data)
    RNA_features.append(RNA_pssm_score)
    # RNA_features.append(calc_motifs_pv(RNA_seq))
    RNA_features.append(generate_one_hot_encoding(RNA_seq))
    RNA_features.append(extract_nucli_features(RNA_seq))
    RNA_features.append(calc_promoter_zones_strength(RNA_seq, RNAp_EDITED_ZONES if rna_type == 'p' else RNAi_EDITED_ZONES))
    RNA_features.append(entropy(RNA_seq))
    RNA_features.append(calculate_dG_and_Tx(RNA_seq)) # 3 features based ution biophysical properties (deltaG)
    RNA_features.append(score_denovo_motifs(RNA_seq))

    RNA_X = pd.concat(RNA_features, axis=1)
    RNA_X.replace(-np.inf, -sys.maxsize, inplace=True)
    RNA_y = RNA_data[TARGET_COLUMN] if cp else None
    return RNA_X, RNA_y


def generate_features_combined(RNA_features: pd.DataFrame, rna_type: str = 'p') -> pd.DataFrame:
    """
    RNA_features: pd.DataFrame, RNA features to concat to the original sequence, for example if we calculate features
     for the original seq of RNAp the RNA_features are for RNAi.
    """
    RNA_seq_original = pd.Series(RNAp_SEQ_ORIGINAL if rna_type == 'p' else RNAi_SEQ_ORIGINAL)
    RNA_df = pd.concat([RNA_seq_original, calc_predicted_promoter_strength(RNA_seq_original)], axis=1)
    RNA_df.rename(columns={RNA_df.columns[0]: 'Promoter Sequence (-35 to +1)'}, inplace=True)
    RNA_X, _ = generate_features(RNA_df, rna_type, cp=False)
    RNA_original_features = pd.DataFrame(np.repeat(RNA_X.values, RNA_features.shape[0], axis=0), columns=RNA_X.columns)
    if rna_type == 'p':
        RNA_X_shared_model = pd.merge(RNA_original_features, RNA_features, left_index=True, right_index=True,
                                      suffixes=('_p', '_i'))
    else:
        RNA_X_shared_model = pd.merge(RNA_features, RNA_original_features, left_index=True, right_index=True,
                                      suffixes=('_p', '_i'))

    RNA_X_shared_model['changed RNA type'] = 0 if rna_type == 'p' else 1  # RNAp will be 0 (and RNAi will be 1)
    return RNA_X_shared_model


def remove_zero_variance_features(X: pd.DataFrame) -> pd.DataFrame:
    zero_variance_cols = X.columns[X.var() == 0]
    return X.drop(zero_variance_cols, axis=1)


# Not in use
def is_high_copy_number(copy_number: 'pd.Series[int]') -> 'pd.Series[int]':
    percentage = 0.2
    n = int(copy_number.shape[0] * percentage)
    high_cp = copy_number.nlargest(n)
    return (copy_number >= high_cp.min()).astype(int)

def equal_bins_data(RNA_df: pd.DataFrame, zero_flag: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    if zero_flag:
        RNA_df['Copy Number'][RNA_df['Copy Number'] < 0] = 0
    RNA_df = RNA_df.sort_values(by='Copy Number')
    num_bins = RNA_df.shape[0] // 15
    bins_series = pd.qcut(RNA_df['Copy Number'], num_bins, labels=False).rename('stratify')
    return RNA_df, bins_series

def split_for_testing(X: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series],
                      stratify_by=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    RNA_X, RNA_X_test, y, y_test = train_test_split(X, y,
                                                    test_size=0.15, random_state=0,
                                                    stratify=stratify_by)
    # RNA_data_train_val = pd.concat([RNA_X, pd.DataFrame(y, columns=[TARGET_COLUMN])], axis=1).\
    #     reset_index(drop=True)
    #
    # RNA_data_test = pd.concat([RNA_X_test, pd.DataFrame(y_test, columns=[TARGET_COLUMN])], axis=1).\
    #     reset_index(drop=True)
    # return RNA_data_train_val, RNA_data_test
    return RNA_X, RNA_X_test


def train_validation_split(X, y, stratify_by: pd.Series,
                           random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state,
                                                      stratify=stratify_by)
    return (X_train.reset_index(drop=True), X_val.reset_index(drop=True),
            y_train.reset_index(drop=True), y_val.reset_index(drop=True))


# def train_validation_test_split(X: pd.DataFrame, y: pd.Series, random_state: int = 0):
#     TARGET_COLUMN_NAME = TARGET_COLUMN
#     y.rename(TARGET_COLUMN_NAME)
#     data = pd.concat((X, y), axis=1)
#     stratify_col = is_high_copy_number(y)
#     data_temp, data_test = split_for_testing(data)
#     data_train, data_validation = train_validation_split(data_temp, stratify_col, random_state)
#     X_train = data_train.drop(TARGET_COLUMN_NAME, axis=1)
#     y_train = data_train[TARGET_COLUMN_NAME]
#     X_valid = data_validation.drop(TARGET_COLUMN_NAME, axis=1)
#     y_valid = data_validation[TARGET_COLUMN_NAME]
#     X_test = data_test.drop(TARGET_COLUMN_NAME, axis=1)
#     y_test = data_test[TARGET_COLUMN_NAME]
#
#     return X_train, X_valid, X_test, y_train, y_valid, y_test



def save_features_df(p=True, i=True, specify_date = False):
    data = {}
    if p:
        print('start generating RNAp features')
        RNAp_data = get_RNAp_data()

        RNAp_data, RNAp_stratify_col = equal_bins_data(RNAp_data)
        RNAp_X = RNAp_data.drop(TARGET_COLUMN, axis=1)
        RNAp_y = RNAp_data[TARGET_COLUMN]

        # Split into train_val and test
        RNAp_data_train_val, RNAp_data_test = split_for_testing(RNAp_X, RNAp_y, stratify_by=RNAp_stratify_col)
        RNAp_stratify_train_val, RNAp_stratify_test = split_for_testing(RNAp_stratify_col, RNAp_y,
                                                                            stratify_by=RNAp_stratify_col)
        RNAp_stratify_train_val = RNAp_stratify_train_val['stratify']
        RNAp_X_train_val = RNAp_data_train_val.drop(TARGET_COLUMN, axis=1)
        RNAp_y_train_val = RNAp_data_train_val[TARGET_COLUMN]

        # Split train_val into train and val
        RNAp_X_data_train, RNAp_X_data_val, RNAp_y_train, RNAp_y_val  = train_validation_split(RNAp_X_train_val, RNAp_y_train_val,
                                                                                               stratify_by=RNAp_stratify_train_val)

        RNAp_data_train = pd.concat([RNAp_X_data_train, RNAp_y_train], axis=1)
        RNAp_data_val = pd.concat([RNAp_X_data_val, RNAp_y_val], axis=1)

        # Generate train split fasta file for high and low copy number motifs
        create_fasta_file(RNAp_data_train, 'p')

        RNAp_X_train, RNAp_y_train = generate_features(RNAp_data_train, rna_type='p')
        RNAp_X_val, RNAp_y_val = generate_features(RNAp_data_val, reference_RNA_data=RNAp_data_train, rna_type='p')
        RNAp_X_test, RNAp_y_test = generate_features(RNAp_data_test, reference_RNA_data=RNAp_data_train_val, rna_type='p')

        final_RNAp_X_train = remove_zero_variance_features(RNAp_X_train)
        final_RNAp_X_val = RNAp_X_val[final_RNAp_X_train.columns]
        final_RNAp_X_test = RNAp_X_test[final_RNAp_X_train.columns]

        data['RNAp_X_train_sequences'] = RNAp_X_data_train
        data['RNAp_X_train'] = final_RNAp_X_train
        data['RNAp_y_train'] = RNAp_y_train

        data['RNAp_X_val_sequences'] = RNAp_X_data_val
        data['RNAp_X_val'] = final_RNAp_X_val
        data['RNAp_y_val'] = RNAp_y_val

        data['RNAp_X_test_sequences'] = RNAp_data_test
        data['RNAp_X_test'] = final_RNAp_X_test
        data['RNAp_y_test'] = RNAp_y_test

        data['RNAp_stratify_train_val'] = RNAp_stratify_train_val
        data['RNAp_stratify_test'] = RNAp_stratify_test

        data['RNAp_X_sequences'] = RNAp_X
        data['RNAp_X'] = pd.concat((final_RNAp_X_train, final_RNAp_X_val, final_RNAp_X_test), ignore_index=True)
        data['RNAp_y'] = pd.concat((RNAp_y_train_val, RNAp_y_test), ignore_index=True)

        filename = 'RNAp_DataFrame_with_features.joblib'

    if i:
        print('start generating RNAi features')
        RNAi_data = get_RNAi_data()


        RNAi_data, RNAi_stratify_col = equal_bins_data(RNAi_data)
        RNAi_X = RNAi_data.drop(TARGET_COLUMN, axis=1)
        RNAi_y = RNAi_data[TARGET_COLUMN]

        # Split into train_val and test
        RNAi_data_train_val, RNAi_data_test = split_for_testing(RNAi_X, RNAi_y, stratify_by=RNAi_stratify_col)
        RNAi_stratify_train_val, RNAi_stratify_test = split_for_testing(RNAi_stratify_col, RNAi_y,
                                                                        stratify_by=RNAi_stratify_col)
        RNAi_stratify_train_val = RNAi_stratify_train_val['stratify']
        RNAi_X_train_val = RNAi_data_train_val.drop(TARGET_COLUMN, axis=1)
        RNAi_y_train_val = RNAi_data_train_val[TARGET_COLUMN]

        # Split train_val into train and val
        RNAi_X_data_train, RNAi_X_data_val, RNAi_y_train, RNAi_y_val = train_validation_split(RNAi_X_train_val,
                                                                                              RNAi_y_train_val,
                                                                                              stratify_by=RNAi_stratify_train_val)

        RNAi_data_train = pd.concat([RNAi_X_data_train, RNAi_y_train], axis=1)
        RNAi_data_val = pd.concat([RNAi_X_data_val, RNAi_y_val], axis=1)

        # Generate train split fasta file for high and low copy number motifs
        create_fasta_file(RNAi_data_train, 'i')

        RNAi_X_train, RNAi_y_train = generate_features(RNAi_data_train, rna_type='i')
        RNAi_X_val, RNAi_y_val = generate_features(RNAi_data_val, reference_RNA_data=RNAi_data_train, rna_type='i')
        RNAi_X_test, RNAi_y_test = generate_features(RNAi_data_test, reference_RNA_data=RNAi_data_train_val,
                                                     rna_type='i')




        RNAi_from_RNAp_feats = pd.read_csv(Path(DATA_PATH, 'rna_p_unique_features.csv'))
        RNAi_from_RNAp_feats_train_val, RNAi_from_RNAp_feats_test = split_for_testing(RNAi_from_RNAp_feats, RNAi_y,
                                                                                      stratify_by=RNAi_stratify_col)

        RNAi_from_RNAp_feats_train_val = RNAi_from_RNAp_feats_train_val.drop(TARGET_COLUMN, axis=1)

        RNAi_from_RNAp_feats_train, RNAi_from_RNAp_feats_val, _ , _ = train_validation_split(RNAi_from_RNAp_feats_train_val,
                                                                                              RNAi_y_train_val,
                                                                                              stratify_by=RNAi_stratify_train_val)

        RNAi_X_train = pd.concat([RNAi_X_train, RNAi_from_RNAp_feats_train.reset_index()], axis=1)
        RNAi_X_val = pd.concat([RNAi_X_val, RNAi_from_RNAp_feats_val.reset_index()], axis=1)
        RNAi_X_test = pd.concat([RNAi_X_test, RNAi_from_RNAp_feats_test.reset_index()],
                                axis=1)


        #####
        # train_shape = RNAi_X_train.shape[0]
        # val_shape = RNAi_X_val.shape[0]
        # RNAi_X_train = pd.concat([RNAi_X_train, RNAi_from_RNAp_feats.iloc[0: train_shape, :]], axis=1)
        # RNAi_X_val = pd.concat([RNAi_X_val, RNAi_from_RNAp_feats.iloc[train_shape: train_shape + val_shape, :].reset_index()], axis=1)
        # RNAi_X_test = pd.concat([RNAi_X_test, RNAi_from_RNAp_feats.iloc[train_shape + val_shape:, :].reset_index()], axis=1)


        final_RNAi_X_train = remove_zero_variance_features(RNAi_X_train)
        final_RNAi_X_val = RNAi_X_val[final_RNAi_X_train.columns]
        final_RNAi_X_test = RNAi_X_test[final_RNAi_X_train.columns]

        data['RNAi_X_train_sequences'] = RNAi_X_data_train
        data['RNAi_X_train'] = final_RNAi_X_train
        data['RNAi_y_train'] = RNAi_y_train

        data['RNAi_X_val_sequences'] = RNAi_X_data_val
        data['RNAi_X_val'] = final_RNAi_X_val
        data['RNAi_y_val'] = RNAi_y_val

        data['RNAi_X_test_sequences'] = RNAi_data_test
        data['RNAi_X_test'] = final_RNAi_X_test
        data['RNAi_y_test'] = RNAi_y_test

        data['RNAi_stratify_train_val'] = RNAi_stratify_train_val
        data['RNAi_stratify_test'] = RNAi_stratify_test

        data['RNAi_X_sequences'] = RNAi_X
        data['RNAi_X'] = pd.concat((final_RNAi_X_train, final_RNAi_X_val, final_RNAi_X_test), ignore_index=True)
        data['RNAi_y'] = pd.concat((RNAi_y_train_val, RNAi_y_test), ignore_index=True)
        filename = 'RNAi_DataFrame_with_features.joblib'
    if specify_date:
        date = datetime.date
        dump(data, Path(DATA_PATH, date.strftime('%m/%d/%Y') + filename))
    else:
        dump(data, Path(DATA_PATH, filename))
    return data


def get_features_df(rna_type, specify_date=False):
    if rna_type == 'p':
        if Path(DATA_PATH, 'RNAp_DataFrame_with_features.joblib').exists():
            data = load(Path(DATA_PATH, 'RNAp_DataFrame_with_features.joblib'))
        else:
            data = save_features_df(p=True, i=False, specify_date=specify_date)
    elif rna_type == 'i':
        if Path(DATA_PATH, 'RNAi_DataFrame_with_features.joblib').exists():
            data = load(Path(DATA_PATH, 'RNAi_DataFrame_with_features.joblib'))
        else:
            data = save_features_df(p=False, i=True, specify_date=specify_date)
    return data


def create_fasta_file(RNA_df, rna_type):
    percentage = 0.15
    n = int(len(RNA_df) * percentage)
    high_cp = RNA_df.nlargest(n, TARGET_COLUMN)['Promoter Sequence (-35 to +1)']
    low_cp = RNA_df.nsmallest(n, TARGET_COLUMN)['Promoter Sequence (-35 to +1)']
    if rna_type == 'p':
        output_file_high = Path(DATA_PATH, RNAp_high_filename)
        output_file_low = Path(DATA_PATH, RNAp_low_filename)
    elif rna_type == 'i':
        output_file_high = Path(DATA_PATH, RNAi_high_filename)
        output_file_low = Path(DATA_PATH, RNAi_low_filename)
    with open(output_file_high, 'w') as file:
        for idx, sequence in high_cp.items():
            file.write(f'>{idx}\n{sequence}\n')
    with open(output_file_low, 'w') as file:
        for idx, sequence in low_cp.items():
            file.write(f'>{idx}\n{sequence}\n')

def check_selective_mode():
    RNAp_data = get_RNAp_data()
    data_sample = RNAp_data.head()
    # X, y = generate_selected_features(data_sample)
    # X, y = generate_selected_features(RNAp_data)
    X, y = generate_features(data_sample)
    print("done")


if __name__ == '__main__':
    save_features_df()
    # check_selective_mode()

