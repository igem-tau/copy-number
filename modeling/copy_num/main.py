from modeling.copy_num.data_prep.post_process import remove_zero_variance_features
from modeling.copy_num.data_prep.pre_process import *
from modeling.copy_num.features.nucleotide_features import *
from modeling.copy_num.analysis.pssm import *
from modeling.copy_num.features.promotor_strength import calc_promoter_zones_strength
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modeling.copy_num.models.lasso import run_lasso
from modeling.copy_num.models.xgboost import run_xgboost
from modeling.copy_num.models.xgboost import converge_randomsearch
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


def prepare_model_data(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0,stratify=X['Hight_or_low'])
    numeric_features = X_train.select_dtypes(include='float64', exclude='int64')

    scaler = StandardScaler()

    scaler_p = scaler.fit(X_train.loc[:, numeric_features.columns])
    X_train.loc[:, numeric_features.columns] = scaler_p.transform(X_train.loc[:, numeric_features.columns])
    X_test.loc[:, numeric_features.columns] = scaler_p.transform(X_test.loc[:, numeric_features.columns])

    return X_train, X_test, y_train, y_test


def model(train_test,data_df: pd.DataFrame, model_name: str, data_name: str,Best_param={}):
    print(f"Running {model_name} for {data_name}")
    X, y = combine_all_features(data_df, x_col="Promoter Sequence", y_col='Copy Number',
                         **{"promotor_strength": data_df['Predicted Promoter Strength (KbT)'],
                            "pssm_score": None,
                            "motifs_pval": None,
                            })

    X_train, X_test, y_train, y_test = prepare_model_data(X, y)

    if train_test=='train':
        for i in range(7):
            [ii,kk]=converge_randomsearch(X_train, X_test, y_train, y_test, num_of_steps=5, nun_iter=7)
    else:

        if model_name == "lasso":
            run_lasso(X_train, X_test, y_train, y_test, data_title=data_name)
        elif model_name == "xgboost":
            run_xgboost(X_train, X_test, y_train, y_test,Best_param)
        else:
            raise Exception(f"No such model: {model_name}")


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


