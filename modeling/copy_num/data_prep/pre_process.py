import pandas as pd
import os
import numpy as np
from modeling.copy_num.consts import *


timepoints_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_3_seq_cnt_p_rna.xlsx")) # priming RNA time points


def get_pRNA_data():
    """
    get RNA_P df, with additional columns
    :return:
    """
    RNAp_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_1_p_rna.xlsx"))  # priming RNA
    RNAp_df["cnt_grw"] = RNAp_df["Final Counts"] / RNAp_df["Initial Counts"]
    shift = abs(RNAp_df["Copy Number"].min()) + 1e-10
    RNAp_df["copy_num_log"] = np.log(RNAp_df["Copy Number"] + shift)
    return RNAp_df


def get_iRNA_data():
    """
    get RNA_I df, with additional columns
    :return:
    """
    RNAi_df = pd.read_excel(os.path.join("..", "..", "data", "copy_num", "sup_data_2_i_rna.xlsx"))  # inhibitory RNA
    RNAi_df["cnt_grw"] = RNAi_df["Final Counts"] / RNAi_df["Initial Counts"]
    return RNAi_df


def get_pRNA_merged_data():
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


def remove_zero_variance_features(X: pd.DataFrame) -> pd.DataFrame:
  zero_variance_cols = X.columns[X.var() == 0]
  return X.drop(zero_variance_cols, axis=1)
