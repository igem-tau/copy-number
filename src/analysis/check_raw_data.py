from src.data_prep.pre_process import get_RNAp_data, get_RNAi_data
import pandas as pd

def reverse_func(x):
    return ((x + 28) / 123) ** (1 / 0.37)

def func(x):
    return 123*x**(0.37) - 28

def create_full_ori(arr):
    return [s[0:6] + 'TCCTTTTTTTCTGCGCG' + s[6:11] + 'TGCTGC' + s[11:] for s in arr]

def raw_data_testing(RNA_df, checking_values_df):
    checking_values_df['ori'] = create_full_ori(checking_values_df['part_sequences'])

    mapping = {sequence: i for i, sequence in enumerate(RNA_df['Promoter Sequence (-35 to +1)'])}

    checking_values_df['order'] = checking_values_df['ori'].map(mapping)
    checking_values_df = checking_values_df.sort_values(by='order').drop(['order', 'Unnamed: 0'], axis=1)
    checking_values_df = checking_values_df.reset_index(drop = True)

    RNA_df['Raw Copy Number'] = reverse_func(RNA_df['Copy Number'])

    print(sum(checking_values_df['values'].round(2) == RNA_df['Raw Copy Number'].round(2)))

if __name__ == '__main__':
    # ssPCN_values = pd.read_csv('ssPCN_values.csv')
    # RNAp_data = get_RNAp_data()
    # raw_data_testing(RNAp_data, ssPCN_values)

    RNAp_data = get_RNAp_data()
    RNAi_data = get_RNAi_data()
    RNAp_data['Raw Copy Number'] = reverse_func(RNAp_data['Copy Number'])
    RNAi_data['Raw Copy Number'] = reverse_func(RNAi_data['Copy Number'])

    RNAp_data.to_csv('RNAp_with_Raw_PCN.csv')
    RNAi_data.to_csv('RNAi_with_Raw_PCN.csv')



