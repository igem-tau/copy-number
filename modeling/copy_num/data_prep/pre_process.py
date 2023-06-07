from joblib import dump, load
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
from modeling.copy_num.consts import *
from modeling.copy_num.features.denovo_motifs import score_denovo_motifs
from modeling.copy_num.features.motifs import calc_motifs_pv
from modeling.copy_num.features.nucleotide_features import generate_one_hot_encoding, generate_df_from_seq, entropy
from modeling.copy_num.features.promotor_strength import calc_promoter_zones_strength, calc_predicted_promoter_strength
from modeling.copy_num.features.pssm_feature import calc_series_pssm_score
from modeling.copy_num.features.delta_G.TX_prediction import calculate_dG_and_Tx
from modeling.copy_num.models.models_functions import is_high_copy_number
import pathlib

DATA_PATH = os.path.join("..", "..", "..", "data")
timepoints_df = pd.read_excel(
    os.path.join("..", "..", "..", "data", "copy_num", "sup_data_3_seq_cnt_p_rna.xlsx"))  # priming RNA time points
PSSM_THRESHOLD_PATH_p = os.path.join("..", "..", "..", "data", "copy_num", f'pssm_threshold_pRNA.pkl')
PSSM_THRESHOLD_PATH_i = os.path.join("..", "..", "..", "data", "copy_num", f'pssm_threshold_iRNA.pkl')


def get_RNAp_data():
    """
    get RNA_P df, with additional columns
    :return:
    """
    RNAp_df = pd.read_excel(os.path.join(DATA_PATH, "copy_num", "sup_data_1_p_rna.xlsx"),
                            names=RNA_DATA_COLUMNS)  # priming RNA
    RNAp_df["cnt_grw"] = RNAp_df["Final Counts"] / RNAp_df["Initial Counts"]
    shift = abs(RNAp_df["Copy Number"].min()) + 1e-10
    RNAp_df["copy_num_log"] = np.log(RNAp_df["Copy Number"] + shift)
    return RNAp_df


def get_RNAi_data():
    """
    get RNA_I df, with additional columns
    :return:
    """
    RNAi_df = pd.read_excel(os.path.join(DATA_PATH, "copy_num", "sup_data_2_i_rna.xlsx"),
                            names=RNA_DATA_COLUMNS)  # inhibitory RNA
    RNAi_df["cnt_grw"] = RNAi_df["Final Counts"] / RNAi_df["Initial Counts"]
    return RNAi_df


def get_RNAp_merged_data():
    """
    get RNA_p df merged with the timepoints and additional columns
    :return:
    """
    RNAp_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_1_p_rna.xlsx"))  # priming RNA
    timepoints_df = pd.read_excel(
        os.path.join("..", "..", "data", "copy_num", "sup_data_3_seq_cnt_p_rna.xlsx"))  # priming RNA time points
    RNAp_df["cnt_grw"] = RNAp_df["Final Counts"] / RNAp_df["Initial Counts"]
    timepoints_df["avg_dup_rate"] = ((timepoints_df["Timepoint 2 Counts"] / timepoints_df["Timepoint 1 Counts"]) +
                                     (timepoints_df["Timepoint 3 Counts"] / timepoints_df["Timepoint 2 Counts"]) +
                                     (timepoints_df["Timepoint 4 Counts"] / timepoints_df["Timepoint 3 Counts"])) / 3
    timepoints_df["log_avg_dup_rate"] = np.log(timepoints_df["avg_dup_rate"])
    timepoints_df["s2e_dup_rate"] = (timepoints_df["Timepoint 4 Counts"] / timepoints_df["Timepoint 1 Counts"])
    timepoints_df.rename(columns={"Promoter Sequence (-35 to +1)": "Promoter Sequence"}, inplace=True)
    merged = pd.merge(RNAp_df, timepoints_df, on="Promoter Sequence")
    return merged


def generate_features(RNA_data: pd.DataFrame, ref_RNA_data: pd.DataFrame = None, type: str = 'p', cp: bool = True, val: bool = False) -> pd.DataFrame:
    RNA_data.reset_index(inplace=True)
    RNA_seq = RNA_data['Promoter Sequence (-35 to +1)']
    RNA_features = []

    RNA_features.append(RNA_data['Predicted Promoter Strength (KbT)'])

    if val:
        RNA_pssm_score = calc_series_pssm_score(RNA_data, ref_RNA_data)
    else:
        RNA_pssm_score = calc_series_pssm_score(RNA_data, RNA_data)
    RNA_features.append(RNA_pssm_score)

    RNA_features.append(calc_motifs_pv(RNA_seq))
    RNA_features.append(generate_one_hot_encoding(RNA_seq))
    RNA_features.append(generate_df_from_seq(RNA_seq))
    RNA_features.append(calc_promoter_zones_strength(RNA_seq, RNAp_EDITED_ZONES if type == 'p' else RNAi_EDITED_ZONES))
    RNA_features.append(entropy(RNA_seq))
    RNA_features.append(calculate_dG_and_Tx(RNA_seq)) # 3 features based on biophysical properties (deltaG)
    RNA_features.append(score_denovo_motifs(RNA_seq))

    RNA_X = pd.concat(RNA_features, axis=1)
    RNA_y = RNA_data['Copy Number'] if cp else None
    return RNA_X, RNA_y


def generate_features_combined(RNA_features: pd.DataFrame, type: str = 'p') -> pd.DataFrame:
    """
    RNA_features: pd.DataFrame, RNA features to concat to the original sequence, for example if we calculate features
     for the original seq of RNAp the RNA_features are for RNAi.
    """
    RNA_seq_original = pd.Series(RNAp_SEQ_ORIGINAL if type == 'p' else RNAi_SEQ_ORIGINAL)
    RNA_df = pd.concat([RNA_seq_original, calc_predicted_promoter_strength(RNA_seq_original)], axis=1)
    RNA_df.rename(columns={RNA_df.columns[0]: 'Promoter Sequence (-35 to +1)'}, inplace=True)
    RNA_X, _ = generate_features(RNA_df, type, cp=False)
    RNA_original_features = pd.DataFrame(np.repeat(RNA_X.values, RNA_features.shape[0], axis=0), columns=RNA_X.columns)
    if type == 'p':
        RNA_X_shared_model = pd.merge(RNA_original_features, RNA_features, left_index=True, right_index=True,
                                      suffixes=('_p', '_i'))
    else:
        RNA_X_shared_model = pd.merge(RNA_features, RNA_original_features, left_index=True, right_index=True,
                                      suffixes=('_p', '_i'))

    RNA_X_shared_model['changed RNA type'] = 0 if type == 'p' else 1  # RNAp will be 0 (and RNAi will be 1)
    return RNA_X_shared_model


def remove_zero_variance_features(X: pd.DataFrame) -> pd.DataFrame:
    zero_variance_cols = X.columns[X.var() == 0]
    return X.drop(zero_variance_cols, axis=1)

def split_for_validation(RNA_data):
    RNA_X, RNA_X_val, y, y_val = train_test_split(RNA_data.drop(columns=['Copy Number']), RNA_data['Copy Number'], test_size=0.15, random_state=0,
                                                        stratify=is_high_copy_number(RNA_data['Copy Number']))
    RNA_data = pd.concat([RNA_X, pd.DataFrame(y, columns=['Copy Number'])], axis = 1)
    RNAp_data_val = pd.concat([RNA_X_val, pd.DataFrame(y_val, columns=['Copy Number'])], axis = 1)
    return RNA_data, RNAp_data_val


def save_features_df():
    RNAp_data = get_RNAp_data()
    RNAi_data = get_RNAi_data()

    RNAp_data, RNAp_data_val = split_for_validation(RNAp_data)
    RNAi_data, RNAi_data_val = split_for_validation(RNAi_data)

    RNAp_X, RNAp_y = generate_features(RNAp_data, type='p', val = False)
    RNAp_X_val, RNAp_y_val = generate_features(RNAp_data_val, RNAp_data, type='p', val=True)
    RNAi_X, RNAi_y = generate_features(RNAi_data, type='i', val = False)
    RNAi_X_val, RNAi_y_val = generate_features(RNAi_data_val, RNAi_data, type='i', val=True)

    RNAp_X_shared_model = generate_features_combined(RNAi_X, type='p')
    RNAi_X_shared_model = generate_features_combined(RNAp_X, type='i')

    if (RNAi_X_shared_model.columns != RNAp_X_shared_model.columns).any():
        raise Exception('the columns in the shared RNAi and RNAp do not match, must be fixed in order to continue')
    else:
        X_shared_model = remove_zero_variance_features(
            pd.concat([RNAp_X_shared_model, RNAi_X_shared_model], axis=0, ignore_index=True))
        Y_shared_model = pd.concat([RNAi_y, RNAp_y], axis=0, ignore_index=True)

    data = {
        'RNAp_X': remove_zero_variance_features(RNAp_X),
        'RNAp_y': RNAp_y,
        'RNAp_X_val': remove_zero_variance_features(RNAp_X_val),
        'RNAp_y_val': RNAp_y_val,
        'RNAi_X': remove_zero_variance_features(RNAi_X),
        'RNAi_y': RNAi_y,
        'RNAi_X_val': remove_zero_variance_features(RNAi_X_val),
        'RNAi_y_val': RNAi_y_val,
        'X_shared': X_shared_model,
        'Y_shared': Y_shared_model
    }
    dump(data, os.path.join(DATA_PATH, 'DataFrames_with_features.joblib'))
    return data


def get_features_df():
    if os.path.exists(os.path.join(DATA_PATH, 'DataFrames_with_features.joblib')):
        data = load(os.path.join(DATA_PATH, 'DataFrames_with_features.joblib'))
    else:
        from modeling.copy_num.features.motifs import calc_motifs_pv
        data = save_features_df()

    return data


if __name__ == '__main__':
    save_features_df()