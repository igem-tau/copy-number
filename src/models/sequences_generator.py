import os.path

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Union
from functools import partial
from pathlib import Path

from src.consts import RNA_DATA_COLUMNS, NUCLEOTIDES, RNAp_SEQ_ORIGINAL, RNAi_SEQ_ORIGINAL, START_INDEX
from src.utils import get_current_file_parent_path
from joblib import dump, load


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')


def sequence_generator(mutation_locations: List[Tuple[int, int]], rna_type: str) -> 'pd.Series[str]':
    if rna_type == 'p':
        template = RNAp_SEQ_ORIGINAL
    elif rna_type == 'i':
        template = RNAi_SEQ_ORIGINAL
    else:
        raise 'Choose a valid rna_type (i or p)'

    def combine_sequence(template: str, mutations: str, mutation_locations: List[Tuple[int, int]]) -> str:

        START_INDEX = -35
        new_seq = template
        current_start = 0
        for start_loc, end_loc in mutation_locations:
            mutation_length = end_loc - start_loc + 1
            new_seq = new_seq[0: start_loc - START_INDEX] + \
                      mutations[current_start: current_start + mutation_length] + \
                      new_seq[end_loc + 1 - START_INDEX:]
            current_start += mutation_length
        return new_seq

    total_mutations = sum([end - start + 1 for start, end in mutation_locations])
    all_mutations = [''.join(mutations) for mutations in itertools.product(NUCLEOTIDES, repeat=total_mutations)]

    return pd.Series(
        [combine_sequence(template, mutations, mutation_locations) for mutations in all_mutations],
        name='Promoter Sequence (-35 to +1)')


# import all supplementary data
# priming RNA
RNAp_df = pd.read_excel(os.path.join(DATA_PATH, 'sup_data_1_p_rna.xlsx'), names=RNA_DATA_COLUMNS)
timepoints_df = pd.read_excel(os.path.join(DATA_PATH, 'sup_data_3_seq_cnt_p_rna.xlsx'))
# inhibitory RNA
RNAi_df = pd.read_excel(os.path.join(DATA_PATH, 'sup_data_2_i_rna.xlsx'), names=RNA_DATA_COLUMNS)



# promoter energy matrix
def get_energy_matrix_for_rna_polymeras() -> pd.DataFrame:
    FIRST_INDEX = -40
    RAW_DATA = np.loadtxt('https://doi.org/10.1371/journal.pcbi.1002811.s003')
    return pd.DataFrame(RAW_DATA, columns=list(NUCLEOTIDES), index=np.arange(
        FIRST_INDEX, FIRST_INDEX + len(RAW_DATA)))


# calculate the energy matrix:
energy_matrix = get_energy_matrix_for_rna_polymeras()
def calc_zone_strength(seq: str, zone: Tuple[int], energy_mat=energy_matrix) -> 'pd.Series[float]':
    seq = seq[:-1]  # delete +1 position
    start_zone, end_zone = zone
    strength = 0

    for i in range(start_zone, end_zone + 1):
        strength += energy_mat.loc[i, seq[i - START_INDEX]]
    return strength

def calc_promoter_zones_strength(seq: 'pd.Series[str]', zones=List[Tuple[int]], energy_mat=energy_matrix) -> pd.DataFrame:
    zones_strength = {}
    for zone in zones:
        strength = seq.apply(
            partial(calc_zone_strength, zone=zone, energy_mat=energy_mat))
        zones_strength[f'{zone} predicted strength'] = strength

    return pd.DataFrame(zones_strength)


def calc_predicted_promoter_strength(seq: Union[str, 'pd.Series[str]']) -> Union[float, 'pd.Series[float]']:
    predicted_promoter_strength = calc_promoter_zones_strength(seq, [(-35,-1)], energy_mat=energy_matrix)['(-35, -1) predicted strength']
    return predicted_promoter_strength.rename('Predicted Promoter Strength (KbT)')

def sequence_df_generator(rna_type = 'p'):
    if rna_type=='p':
        if Path(DATA_PATH, 'RNAp_Generated_Sequences.joblib').exists():
            data = load(Path(DATA_PATH, 'RNAp_Generated_Sequences.joblib'))
        else:
            generated_RNAp_seq = sequence_generator([(-33, -30), (-11, -8), (0, 0)], rna_type)
            generated_RNAp_promoter_strength = calc_predicted_promoter_strength(generated_RNAp_seq)
            data = pd.concat([generated_RNAp_seq, generated_RNAp_promoter_strength],
                                      axis=1)
            dump(data, Path(DATA_PATH, 'RNAp_Generated_Sequences.joblib'))
    elif rna_type=='i':
        if Path(DATA_PATH, 'RNAi_Generated_Sequences.joblib').exists():
            data = load(Path(DATA_PATH, 'RNAi_Generated_Sequences.joblib'))
        else:
            generated_RNAp_seq = sequence_generator([(-33, -30), (-10, -7), (0, 0)], rna_type)
            generated_RNAp_promoter_strength = calc_predicted_promoter_strength(generated_RNAp_seq)
            data = pd.concat([generated_RNAp_seq, generated_RNAp_promoter_strength],
                                      axis=1)
            dump(data, Path(DATA_PATH, 'RNAi_Generated_Sequences.joblib'))
    return data
