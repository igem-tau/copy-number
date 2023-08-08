import numpy as np
import pandas as pd
import itertools
from functools import partial
from typing import Dict, List, Tuple, Union


def sequence_generator(mutation_locations: List[Tuple[int, int]], rna_type: str) -> 'pd.Series[str]':
    if rna_type =='p':
        template = 'TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT'
    elif rna_type == 'i':
        template = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
    else:
        raise 'Choose a valid rna_type (i or p)'

    NUCLEOTIDES = 'ACGT'
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


a=5