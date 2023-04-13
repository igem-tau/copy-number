from modeling.copy_num.consts import *
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Tuple
from functools import partial


def get_energy_matrix_for_rna_polymeras() -> pd.DataFrame:
    """
    from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002811
        data link: https://doi.org/10.1371/journal.pcbi.1002811.s003
        Energy matrix for RNAP in kT. Inferred from an experiment done in
        TK310 with no supplemental cAMP (and hence, no CRP present in the
        cells). The matrix covers base pairs [-41:-1] where 0 denotes the
        transcription start site. Each row corresponds to a given position;
        each column corresponds to a value for that base pair. The columns
        are ordered [A,C,G,T].
    :return:
    """
    FIRST_INDEX = -40
    RAW_DATA = np.loadtxt('https://doi.org/10.1371/journal.pcbi.1002811.s003')
    return pd.DataFrame(RAW_DATA, columns=list(NUCLEOTIDES), index=np.arange(FIRST_INDEX, FIRST_INDEX + len(RAW_DATA)))


def plot_energy_matrix():
    # plot the energy matrix
    fig = sns.heatmap(
        get_energy_matrix_for_rna_polymeras().T,
        cmap='jet',
        square=True,
        yticklabels=['A', 'C', 'G', 'T'],
        xticklabels=range(-41, 0),
        cbar_kws={'label': r'$Energy [K_{B}T]$'}
    )
    _ = fig.set_title('Energy Matrix for promoter strength')
    _ = fig.set_xlabel('position')


def calc_promoter_zones_strength(seq: 'pd.Series[str]', zones=List[Tuple[int]]) -> pd.DataFrame:
    energy_matrix = get_energy_matrix_for_rna_polymeras()

    def calc_zone_strength(seq:str, zone: Tuple[int], energy_matrix) -> 'pd.Series[float]':
        seq = seq[:-1] # delete +1 position
        start_zone, end_zone = zone
        strength = 0

        for i in range(start_zone, end_zone + 1):
            strength += energy_matrix.loc[i, seq[i - START_INDEX]]
        return strength

    zones_strength = {}
    for zone in zones:
        strength = seq.apply(
            partial(calc_zone_strength, zone=zone, energy_matrix=energy_matrix)
        )
        zones_strength[f'{zone} predicted strength'] = strength

    return pd.DataFrame(zones_strength)


def create_hight_or_low_features(X: pd.DataFrame, Type_of_thresh) -> pd.DataFrame:
    l=['mean', 'median']
    if Type_of_thresh not in l:
        while True:
            u_in = input("Please enter a precentage between 0 and 100: ")
            try:
                number = float(u_in)
                if 0 <= number <= 100:
                    Threshold=max(X["Copy Number"])*number/100
                    print(f'Threshold was selected to be {Threshold} ({number}%)')
                    break
                else:
                    print("The number must be between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    elif Type_of_thresh=='mean':
        Threshold = X["Copy Number"].mean()
    elif Type_of_thresh=='med':
        Threshold = X["Copy Number"].median()

    return(pd.DataFrame(np.array([X["Copy Number"] <= Threshold]).T.astype(int),columns=['Hight_or_low']))

