import pandas as pd
import numpy as np
from Bio import motifs
from functools import partial
from joblib import dump, load
import os


PSSM_THRESHOLD_PATH = 'pssm_threshold.pkl'

def set_pssm_thresholds(seq: 'pd.Series[str]') -> None:
    percentage = 0.2
    n = int(len(seq) * percentage)
    high_cp = seq.nlargest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
    low_cp = seq.nsmallest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
    pssm_threshold = {'high': high_cp.min(), 'low': low_cp.max()}
    dump(pssm_threshold, PSSM_THRESHOLD_PATH)


# a function to calculate is a copy-number part of the top 20% in relation to all the others
def is_high_copy_number(copy_number: 'pd.Series[int]') -> 'pd.Series[int]':
    if not os.exist(PSSM_THRESHOLD_PATH):
        raise Exception('you need to set pssm thresholds first')
    pssm_threshold = load(PSSM_THRESHOLD_PATH)
    return copy_number >= pssm_threshold['high']


# generate the scoring matrix (typically from the highest 20%)
def calc_pssm_matrix(data: 'pd.Series[str]') -> pd.DataFrame:
    cp_motifs = motifs.create(data.tolist())
    pwm = pd.DataFrame(cp_motifs.pwm)
    return (np.log2(pwm) * pwm).fillna(0)


def calc_pssm_score(seq: str, pssm: pd.DataFrame) -> float:
    score = 0
    for i, nt in enumerate(seq):
        score += pssm.loc[i, nt]
    return score


def calc_series_pssm_score(seq: 'pd.Series[str]', pssm=None) -> 'pd.Series[float]':
    if pssm in None:
        top_20_percent = seq.loc[is_high_copy_number(seq)]
        pssm = calc_pssm_matrix(top_20_percent)
    pssm_scores = seq.apply(partial(calc_pssm_score, pssm=pssm))
    return pssm_scores.rename('pssm_score')