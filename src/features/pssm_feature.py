from Bio import motifs
from src.consts import *
from functools import partial
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import get_current_file_parent_path


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
PSSM_THRESHOLD_PATH_p = Path(DATA_PATH, 'pssm_threshold_pRNA.pkl')
PSSM_THRESHOLD_PATH_i = Path(DATA_PATH, 'pssm_threshold_iRNA.pkl')


# TODO - check redundancy - not in use
# a function to calculate is a copy-number part of the top 20% in relation to all the others
def is_high_copy_number(copy_number: 'pd.Series[int]', rna_type: str = 'p') -> 'pd.Series[int]':
    if rna_type == 'p':
        PSSM_THRESHOLD_PATH = PSSM_THRESHOLD_PATH_p
    else:
        PSSM_THRESHOLD_PATH = PSSM_THRESHOLD_PATH_i
    if not PSSM_THRESHOLD_PATH.exists():
        raise Exception('you need to set pssm thresholds first - run "pssm_feature.py"')
    pssm_data = load(PSSM_THRESHOLD_PATH)
    return copy_number >= pssm_data['high']


# generate the scoring matrix (typically from the highest 20%)
def calc_pssm_matrix(data: 'pd.Series[str]') -> pd.DataFrame:
    cp_motifs = motifs.create(data.tolist())
    pwm = pd.DataFrame(cp_motifs.pwm)
    return (np.log2(pwm/0.25)).fillna(0)



def calc_pssm_score(seq: str, pssm: pd.DataFrame) -> float:
    score = 0
    for i, nt in enumerate(seq):
        score += pssm.loc[i, nt]
    return score


def calc_series_pssm_score(seq: 'pd.Series[str]', ref_seq: 'pd.Series[str]', pssm=None) -> 'pd.Series[float]':
    if pssm is None:
        # top_20_percent = ref_seq.loc[is_high_copy_number(ref_seq)]
        # pssm = calc_pssm_matrix(top_20_percent)
        pssm = set_pssm_thresholds(ref_seq)
    pssm_scores = seq['Promoter Sequence (-35 to +1)'].apply(partial(calc_pssm_score, pssm=pssm))
    return pssm_scores.rename('pssm_score')

def set_pssm_thresholds(RNA_df: pd.DataFrame, rna_type:str = 'p') -> None:
    percentage = 0.2
    n = int(len(RNA_df) * percentage)
    high_cp = RNA_df.nlargest(n, TARGET_COLUMN)['Promoter Sequence (-35 to +1)']
    low_cp = RNA_df.nsmallest(n, TARGET_COLUMN)['Promoter Sequence (-35 to +1)']
    RNA_pssm = calc_pssm_matrix(high_cp)
    # pssm_threshold = {'high': high_cp.min(), 'low': low_cp.max(), 'pssm_matrix': RNA_pssm}
    # if rna_type == 'p':
    #     PSSM_THRESHOLD_PATH = PSSM_THRESHOLD_PATH_p
    # else:
    #     PSSM_THRESHOLD_PATH = PSSM_THRESHOLD_PATH_i
    # dump(pssm_threshold, PSSM_THRESHOLD_PATH, compress=True)
    return RNA_pssm


if __name__ == '__main__':
    from src.data_prep.pre_process import get_RNAp_data, get_RNAi_data
    RNA_type = 'p'
    RNAp_df = get_RNAp_data()
    set_pssm_thresholds(RNAp_df, RNA_type)

    RNA_type = 'i'
    RNAi_df = get_RNAi_data()
    set_pssm_thresholds(RNAi_df, RNA_type)
