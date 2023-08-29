from src.features.rna_structure import get_avg_mfe_per_position, \
    get_dist_from_orig_alpha_beta, get_alpha_area_match_ratio

from src.rna_p.data_prep import load_data, get_wild_type
from src.data_prep.pre_process import get_RNAi_prom_RNAp
from src.features.rna_structure import make_rna_features


import pandas as pd
from src.features.delta_G.TX_prediction import calculate_dG_and_Tx


SEQ_COLUMN_NAME = "RNAp_seq"


def check_prom_strength():
    df = pd.read_csv(r"C:\Users\User1\IGEM\code\copy-number\data\rna_p_data.csv")
    df = df.sort_values(by="Copy Number")
    pdf = df["Promoter Sequence (-35 to +1)"]
    calculate_dG_and_Tx(pdf)


def get_rna_features():
    df = get_RNAi_prom_RNAp()
    rna_features = make_rna_features(df["RNAp_seq"])





# def generate_features():
#     df = load_data()
#
#     # Todo: for test purpose (remember to remove)
#     # df = df.head(5)
#
#     df["entropy"] = entropy(df[SEQ_COLUMN_NAME])
#     # Todo: check how to use it as a feature
#     # res = get_avg_mfe_per_position(df, SEQ_COLUMN_NAME)
#
#     wt = get_wild_type()
#     df = add_mut_in_pos(df, SEQ_COLUMN_NAME, wt)
#     df = add_rna_mfe_diff(df, SEQ_COLUMN_NAME, wt)
#     df = rna_topo_dist(df, SEQ_COLUMN_NAME, wt)
#
#     print("ok")


if __name__ == '__main__':
    # generate_features()
    check_prom_strength()