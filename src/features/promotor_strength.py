from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from src.consts import *
from src.utils import get_current_file_parent_path, is_feature_selected
from typing import List, Tuple, Union, Optional


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
    CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
    RAW_DATA = np.loadtxt(Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'pcbi.1002811.s003.txt'))
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

    fig.set_title('Energy Matrix for promoter strength')
    fig.set_xlabel('position')


def calc_promoter_zones_strength(seq: 'pd.Series[str]', zones: List[Tuple[int, int]],
                                 selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    energy_matrix = get_energy_matrix_for_rna_polymeras()

    def calc_zone_strength(seq: str, zone: Tuple[int, int], energy_matrix_) -> float:
        seq = seq[:-1]  # delete +1 position
        start_zone, end_zone = zone
        strength = 0.0

        for i in range(start_zone, end_zone + 1):
            strength += energy_matrix_.loc[i, seq[i - START_INDEX]]
        return strength

    print("Running: calc_promoter_zones_strength")
    zones_strength = {}
    for zone in zones:
        zone_name = f'{zone} predicted strength'
        if is_feature_selected(zone_name, selected_features):
            strength = seq.apply(partial(calc_zone_strength, zone=zone, energy_matrix_=energy_matrix))
            zones_strength[zone_name] = strength

    return pd.DataFrame(zones_strength)


def calc_predicted_promoter_strength(seq: Union[str, 'pd.Series[str]']) -> Union[float, 'pd.Series[float]']:
    predicted_promoter_strength = calc_promoter_zones_strength(seq, [(-35, -1)])['(-35, -1) predicted strength']
    return predicted_promoter_strength.rename('Predicted Promoter Strength (KbT)')


if __name__ == '__main__':
    seq = pd.Series(["TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT",
                     "TTGAAATCCTTTTTTTCTGCGCGTAATCTTTTGCTT",
                     "TAGCGATCCTTTTTTTCTGCCGGTAATCTGCTGCTT",
                     "GTTAGATCCTTTTTTTCTGCGCGTTATACACTGCTT",
                     "TTAGAATCGCCTTTTTCTGCGCGTAATCTGCTAAAT"])
    zones = [(-33, -30), (-11, -8)]
    calc_promoter_zones_strength(seq, zones, None)
