from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from modeling.copy_num.data_prep.pre_process import *


def show_corr_between_2_columns(df: pd.DataFrame, x_column_name: str, y_column_name: str):
    corr_coef = df[x_column_name].corr(df[y_column_name])
    pearson_corr_coef, pearson_p_value = pearsonr(df[x_column_name], df[y_column_name])
    spearman_corr_coef, spearman_p_value = pearsonr(df[x_column_name], df[y_column_name])
    sns.scatterplot(x=x_column_name, y=y_column_name, data=df)
    plt.title('Correlation coefficient: {:.2f}\n'
              'Pearson corr coef: {:.2f} p-value: {:.2e} ({:.2f})\n'
              'Spearman corr coef: {:.2f} p-value: {:.2e} ({:.2f})'.format(corr_coef,
                                                                           pearson_corr_coef,
                                                                           pearson_p_value,
                                                                           float(pearson_p_value),
                                                                           spearman_corr_coef,
                                                                           spearman_p_value,
                                                                           float(spearman_p_value)))
    sns.regplot(x=x_column_name, y=y_column_name, data=df)
    plt.show()


def check_some_correlations():
    p_rna_df = get_pRNA_data()

    # corr between promotor strength and copy num
    show_corr_between_2_columns(p_rna_df, "cnt_grw", "Copy Number")
    show_corr_between_2_columns(p_rna_df, "Predicted Promoter Strength (KbT)", "Copy Number")
    show_corr_between_2_columns(p_rna_df, "Predicted Promoter Strength (KbT)", "cnt_grw")

    # corr between promotor strength and log(copy num)
    show_corr_between_2_columns(p_rna_df, "cnt_grw", "copy_num_log")
    show_corr_between_2_columns(p_rna_df, "Predicted Promoter Strength (KbT)", "copy_num_log")
    show_corr_between_2_columns(p_rna_df, "Predicted Promoter Strength (KbT)", "cnt_grw")

    p_rna_merged_df = get_pRNA_merged_data()
    show_corr_between_2_columns(p_rna_merged_df, "Predicted Promoter Strength (KbT)", "log_avg_dup_rate")


def calc_correlations_with_copynumber(RNA_features_df,RNA_copynumber_df):
    pearson_corr=[]
    pearson_pv=[]
    spearman_corr=[]
    spearman_pv=[]
    for i in range(RNA_features_df.shape[1]):
        Pcorr,Ppv=pearsonr(RNA_features_df.iloc[:,i],RNA_copynumber_df)
        Scorr,Spv=spearmanr(RNA_features_df.iloc[:,i],RNA_copynumber_df)
        pearson_corr.append(Pcorr)
        pearson_pv.append(Ppv)
        spearman_corr.append(Scorr)
        spearman_pv.append(Spv)
    data_with_pv={
        'Pearson Correlation': pearson_corr,
        'Pearson pv': pearson_pv,
        'Spearman Correlation': spearman_corr,
        'Spearman pv':spearman_pv
    }
    # data={'Pearson Correlation':pearson_corr,'Spearman Correlation':spearman_corr}
    # return pd.DataFrame(data,index=RNA_features_df.columns).T
    return pd.DataFrame(data_with_pv,index=RNA_features_df.columns).T