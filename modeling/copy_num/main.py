from modeling.copy_num.data_prep.pre_process import *
from modeling.copy_num.features.nucleotide_features import *
from modeling.copy_num.analysis.pssm import *
from modeling.copy_num.features.promotor_strength import calc_promoter_zones_strength
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modeling.copy_num.models.lasso import run_lasso
from modeling.copy_num.models.xgboost_model import run_xgboost
from modeling.copy_num.models.xgboost_model import converge_randomsearch
import warnings


def show_nucliotide_distrib_per_group():
    """
    Here we can see weblogo of the relevant sequence per buckets
    :return:
    """
    p_rna_df = get_RNAp_data()
    buckets = split_by_buckets(p_rna_df, 4, "Copy Number")
    pssm_per_bucket = get_pssm_using_motifs_for_buckets(buckets, show=False)


def combine_all_features(df: pd.DataFrame, x_col: str, y_col: str, **kwargs):
    src_data = df[x_col]
    features = []

    for i, v in kwargs.items():
        if v is not None:
            features.append(v)
    # features.append(df['Predicted Promoter Strength (KbT)'])

    # RNAp_features.append(RNAp_pssm_score)
    # RNAp_features.append(calc_motifs_pv(RNAp_seq))

    features.append(generate_one_hot_encoding(src_data))
    features.append(generate_df_from_seq(src_data))
    features.append(calc_promoter_zones_strength(src_data, RNAp_EDITED_ZONES))

    X_temp = pd.concat(features, axis=1)
    X = remove_zero_variance_features(X_temp)
    y = df[y_col]
    return X, y





def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # modeling for p-RNA
    train_test=''
    pRNA_df = get_RNAp_data()
    # model(train_test,pRNA_df, model_name="lasso", data_name="p RNA")
    # model(train_test,pRNA_df, model_name="xgboost", data_name="p RNA")

    # modeling for i-RNA
    iRNA_df = get_RNAi_data()
    # model(train_test,iRNA_df, model_name="lasso", data_name="i RNA")
    model(train_test,iRNA_df, model_name="xgboost", data_name="i RNA")

    # Todo: add modeling for combined data


if __name__ == '__main__':
    main()


