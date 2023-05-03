from joblib import dump
import pandas as pd
import os
import numpy as np
from modeling.copy_num.consts import *
from modeling.copy_num.features.motifs import calc_motifs_pv
from modeling.copy_num.features.nucleotide_features import generate_one_hot_encoding, generate_df_from_seq
from modeling.copy_num.features.promotor_strength import calc_promoter_zones_strength
from modeling.copy_num.features.pssm_feature import calc_series_pssm_score

timepoints_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_3_seq_cnt_p_rna.xlsx")) # priming RNA time points


def get_RNAp_data():
    """
    get RNA_P df, with additional columns
    :return:
    """
    RNAp_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_1_p_rna.xlsx"), names=RNA_DATA_COLUMNS)  # priming RNA
    RNAp_df["cnt_grw"] = RNAp_df["Final Counts"] / RNAp_df["Initial Counts"]
    shift = abs(RNAp_df["Copy Number"].min()) + 1e-10
    RNAp_df["copy_num_log"] = np.log(RNAp_df["Copy Number"] + shift)
    return RNAp_df


def get_RNAi_data():
    """
    get RNA_I df, with additional columns
    :return:
    """
    RNAi_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_2_i_rna.xlsx"), names=RNA_DATA_COLUMNS)  # inhibitory RNA
    RNAi_df["cnt_grw"] = RNAi_df["Final Counts"] / RNAi_df["Initial Counts"]
    return RNAi_df


def get_RNAp_merged_data():
    """
    get RNA_p df merged with the timepoints and additional columns
    :return:
    """
    RNAp_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_1_p_rna.xlsx"))  # priming RNA
    timepoints_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_3_seq_cnt_p_rna.xlsx"))  # priming RNA time points
    RNAp_df["cnt_grw"] = RNAp_df["Final Counts"] / RNAp_df["Initial Counts"]
    timepoints_df["avg_dup_rate"] = ((timepoints_df["Timepoint 2 Counts"] / timepoints_df["Timepoint 1 Counts"]) +
                                     (timepoints_df["Timepoint 3 Counts"] / timepoints_df["Timepoint 2 Counts"]) +
                                     (timepoints_df["Timepoint 4 Counts"] / timepoints_df["Timepoint 3 Counts"])) / 3
    timepoints_df["log_avg_dup_rate"] = np.log(timepoints_df["avg_dup_rate"])
    timepoints_df["s2e_dup_rate"] = (timepoints_df["Timepoint 4 Counts"] / timepoints_df["Timepoint 1 Counts"])
    timepoints_df.rename(columns={"Promoter Sequence (-35 to +1)": "Promoter Sequence"}, inplace=True)
    merged = pd.merge(RNAp_df, timepoints_df, on="Promoter Sequence")
    return merged

def generate_features(RNA_data: pd.DataFrame, type:str='p', cp:bool=False) -> pd.DataFrame:
    RNA_seq = RNA_data['Promoter Sequence (-35 to +1)']
    RNA_features = []

    RNA_features.append(RNA_data['Predicted Promoter Strength (KbT)'])
    RNA_pssm_score = calc_series_pssm_score(RNA_data['Promoter Sequence (-35 to +1)'])
    RNA_features.append(RNA_pssm_score)
    RNA_features.append(calc_motifs_pv(RNA_seq))
    RNA_features.append(generate_one_hot_encoding(RNA_seq))
    RNA_features.append(generate_df_from_seq(RNA_seq))
    RNA_features.append(calc_promoter_zones_strength(RNA_seq, RNAp_EDITED_ZONES if type == 'p' else RNAi_EDITED_ZONES))

    RNA_X = pd.concat(RNA_features, axis=1)
    RNA_y = RNA_data['Copy Number'] if cp else None
    return RNA_X, RNA_y

def generate_features_combined(seq: 'pd.Series[str]', type:str='p', RNA_features: 'pd.DataFrame') -> pd.DataFrame:

    RNA_seq_original = pd.Series(RNAi_SEQ_ORIGINAL)
    RNA_X, _ = generate_features(RNA_seq_original, type, cp=False)

    RNA_original_features = pd.DataFrame(np.repeat(RNA_X.values, RNA_X.shape[0], axis=0), columns=RNA_X.columns)

    RNA_X_shared_model = pd.merge(RNA_features, RNA_original_features, left_index=True, right_index=True, suffixes=('_p', '_i'))
    RNA_X_shared_model['changed RNA type'] = 0 if type == 'p' else 1  # RNAp will be 0 (and RNAi will be 1)


def save_features_df():
    RNAp_data = get_RNAp_data()
    RNAi_data = get_RNAi_data()

    RNAp_X, RNAp_y = generate_features(RNAp_data)
    RNAi_X, RNAi_y = generate_features(RNAi_data)

    data = {
        'RNAp_X': RNAp_X,
        'RNAp_y': RNAp_y,
        'RNAi_X': RNAi_X,
        'RNAi_y': RNAi_y,
        'X_shared': X_shared_model,
        'Y_shared': Y_shared_model
    }
    dump(data, 'DataFrames_with_features.joblib')