from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from src.data_prep.pre_process import get_features_df
from src.utils import get_current_file_parent_path
from typing import Dict

CORRELATIONS_METHODS = ['pearson', 'kendall', 'spearman']
CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


def calc_MIC_pv(original_mine: 'MINE', RNA_Series: 'pd.Series', RNA_copynumber: 'pd.Series') -> dict:
    mics = []
    for i in range(1000):
        new_mine = MINE(alpha=0.6, c=15, est='mic_approx')
        permuted_RNA_Series = np.random.permutation(RNA_Series)
        permuted_RNA_copynumber = np.random.permutation(RNA_copynumber)
        new_mine.compute_score(permuted_RNA_Series, permuted_RNA_copynumber)
        mics.append(new_mine.mic())
    original_mic = original_mine.mic()
    pv = np.sum(np.asarray(mics) >= original_mic) / len(mics)
    return {'mics': mics, 'pv': pv}

def calc_correlations_with_copynumber(RNA_features_df:'pd.DataFrame',RNA_copynumber_df:'pd.Series') -> dict:
    pearson_corr=[]
    pearson_pv=[]
    spearman_corr=[]
    spearman_pv=[]
    mic=[]
    mic_pv=[]
    mics_permutation_data={}
    for i in range(RNA_features_df.shape[1]):
        RNA_series=RNA_features_df.iloc[:,i]
        Pcorr,Ppv=pearsonr(RNA_series,RNA_copynumber_df)
        Scorr,Spv=spearmanr(RNA_series,RNA_copynumber_df)
        mine = MINE(alpha=0.6, c=15, est='mic_approx')
        mine.compute_score(RNA_series,RNA_copynumber_df)
        MIC_data=calc_MIC_pv(mine,RNA_series,RNA_copynumber_df)
        Mpv=MIC_data['pv']
        mics_permutation_data[RNA_series.name]=MIC_data['mics']
        pearson_corr.append(Pcorr)
        pearson_pv.append(Ppv)
        spearman_corr.append(Scorr)
        spearman_pv.append(Spv)
        mic.append(mine.mic())
        mic_pv.append(Mpv)
        if i%100==0:
          print(i, ' done')
    data={
        'Pearson Correlation': pearson_corr,
        'Pearson pv': pearson_pv,
        'Spearman Correlation': spearman_corr,
        'Spearman pv':spearman_pv,
        'MIC': mic,
        'MIC pv': mic_pv
    }
    correlation_df=pd.DataFrame(data,index=RNA_features_df.columns).T
    permutations_df=pd.DataFrame(mics_permutation_data)
    final_data={'Correlation df':correlation_df,'Permutations df':permutations_df}
    return final_data

def calc_mic(x: 'pd.Series[float]', y: 'pd.Series[float]') -> Dict[str, float]:
  mine = MINE()
  mine.compute_score(x, y)
  correlation_scores = {}
  correlation_scores['Maximal Information Coefficient (MIC or MIC_e)'] = [mine.mic()]
  correlation_scores['Maximum Asymmetry Score (MAS)'] = [mine.mas()]
  correlation_scores['Maximum Edge Value (MEV)'] = [mine.mev()]
  return correlation_scores

def PSSM_corr_plot(RNA_pssm_score: pd.Series, copy_number: pd.Series, title:str):
    RNAp_pssm_correlations = {}
    for corr_method in CORRELATIONS_METHODS:
        corr_score = RNA_pssm_score.corr(copy_number, method=corr_method)
        RNAp_pssm_correlations[corr_method] = [corr_score]
    RNAp_pssm_correlations.update(calc_mic(RNA_pssm_score, copy_number))
    plt.figure()
    pd.DataFrame(RNAp_pssm_correlations).plot.bar(figsize=(15, 5))
    plt.title(title)

def promoter_strength_plot(RNA_X, RNA_y, RNA_correlations_df, rna_type:str= 'p'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(RNA_X['Predicted Promoter Strength (KbT)'], RNA_y)
    ax.set_xlabel(f'RNA{rna_type} Promoter Strength')
    ax.set_ylabel(f'RNA{rna_type} Predicted Copy Number')
    ax.set_title(f'RNA{rna_type} Promoter Strength correlation with copy number')
    ax.text(0.8, 0.8, 'MIC=%.4f' % RNA_correlations_df.loc['MIC', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    ax.text(0.8, 0.7, 'MIC pv=%.4f' % RNA_correlations_df.loc['MIC pv', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    ax.text(0.8, 0.6,
            'Pearson=%.4f' % RNA_correlations_df.loc['Pearson Correlation', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    ax.text(0.8, 0.5, 'Pearson pv=%.4f' % RNA_correlations_df.loc['Pearson pv', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    ax.text(0.8, 0.4,
            'Spearman=%.4f' % RNA_correlations_df.loc['Spearman Correlation', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    ax.text(0.8, 0.3, 'Spearman pv=%.4f' % RNA_correlations_df.loc['Spearman pv', 'Predicted Promoter Strength (KbT)'],
            transform=ax.transAxes)
    plt.show()

if __name__ == '__main__':
    data = get_features_df()

    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    # PSSM
    PSSM_corr_plot(RNAp_X['pssm_score'], RNAp_y, 'RNAp PSSM')
    PSSM_corr_plot(RNAi_X['pssm_score'], RNAi_y, 'RNAi PSSM')

    # other features
    SAVED_CORRELATIONS_PATH = Path(DATA_PATH, 'correlations_DataFrames.joblib')
    if SAVED_CORRELATIONS_PATH.exists():
        data = load(SAVED_CORRELATIONS_PATH)
        RNAp_correlations_df = data['RNAp_correlations']
        RNAp_permutations_df = data['RNAp_permutations']
        RNAi_correlations_df = data['RNAi_correlations']
        RNAi_permutations_df = data['RNAi_permutations']
    else:
        data_i = calc_correlations_with_copynumber(RNAi_X, RNAi_y)
        data_p = calc_correlations_with_copynumber(RNAp_X, RNAp_y)
        data = {
            'RNAp_correlations': data_p['Correlation df'],
            'RNAp_permutations': data_p['Permutations df'],
            'RNAi_correlations': data_i['Correlation df'],
            'RNAi_permutations': data_i['Permutations df'],
        }
        dump(data, SAVED_CORRELATIONS_PATH, compress=True)
        RNAp_correlations_df = data_p['Correlation df']
        RNAp_permutations_df = data_p['Permutations df']
        RNAi_correlations_df = data_i['Correlation df']
        RNAi_permutations_df = data_i['Permutations df']

    # Display correlations
    # promoter strength:
    promoter_strength_plot(RNAp_X, RNAp_y, RNAp_correlations_df, rna_type='p')
    promoter_strength_plot(RNAi_X, RNAi_y, RNAi_correlations_df, rna_type='i')

