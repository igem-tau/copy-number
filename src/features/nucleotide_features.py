from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp
from datetime import datetime
from dnacurve import CurvedDNA
from itertools import product
import math
import pandas as pd
from src.consts import *
from src.utils import get_selected_features
import subprocess
from typing import List, Dict, Tuple, Union
from tqdm import tqdm


def filtered_one_hot_encoding(seq: pd.Series) -> pd.DataFrame:
    def encode(single_nucleotid: pd.Series, index: int = None) -> pd.DataFrame:
        columns = {}
        for nucleotide in NUCLEOTIDES:
            column_name = f'{nucleotide}_{index}'
            if column_name in selected_features:
                columns[column_name] = (single_nucleotid == nucleotide).astype(int)

        if columns:
            encoded = pd.concat(columns, axis=1)
            return encoded
        return pd.DataFrame()

    selected_features = get_selected_features()
    full_encoding = []
    for current_nucleotide_index in tqdm(range(PROMOTER_LENGTH)):
        current_nucleotide_encoding = encode(seq.str[current_nucleotide_index], current_nucleotide_index + START_INDEX)
        full_encoding.append(current_nucleotide_encoding)

    return pd.concat(full_encoding, axis=1)


def generate_one_hot_encoding(seq: pd.Series) -> pd.DataFrame:
    def encode(single_nucleotid: pd.Series, index: int = None) -> pd.DataFrame:
        columns = []
        for nucleotide in NUCLEOTIDES:
            columns.append((single_nucleotid == nucleotide).astype(int))
        encoded = pd.concat(columns, axis=1)
        if index is not None:
            columns = [f'{nucleotide}_{index}' for nucleotide in NUCLEOTIDES]
        else:
            columns = list(NUCLEOTIDES)
        encoded.columns = columns
        return encoded

    if USE_SELECTED_FEATURES["selective"]:
        return filtered_one_hot_encoding(seq)

    full_encoding = []
    for current_nucleotide_index in range(PROMOTER_LENGTH):
        current_nucleotide_encoding = encode(seq.str[current_nucleotide_index], current_nucleotide_index + START_INDEX)
        full_encoding.append(current_nucleotide_encoding)

    return pd.concat(full_encoding, axis=1)


m2 = list(product(NUCLEOTIDES, repeat=2))
m3 = list(product(NUCLEOTIDES, repeat=3))
m4 = list(product(NUCLEOTIDES, repeat=4))
m5 = list(product(NUCLEOTIDES, repeat=5))
k_gap = 2
k_tuple = 2

selected_features = get_selected_features() if USE_SELECTED_FEATURES["selective"] else None


def kmers(seq: str, k: int) -> List[int]:
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


def pseudo_knc(sequences: 'pd.Series[str]', k: int) -> pd.DataFrame:
    ### k-mer ###
    ### A, AA, AAA

    d = {}
    bio_sequences = sequences.apply(Seq)

    if USE_SELECTED_FEATURES["selective"]:
        for i in tqdm(range(1, k + 1)):
            v = list(product(NUCLEOTIDES, repeat=i))
            for j in v:
                search_seq = ''.join(j)
                key = f'{search_seq}_count'
                if key in selected_features:
                    res = bio_sequences.apply(lambda sequence: sequence.count_overlap(search_seq) / (len(sequence) - len(j) + 1))
                    d[key] = res

        return pd.DataFrame(d)

    for i in tqdm(range(1, k + 1)):
        v = list(product(NUCLEOTIDES, repeat=i))
        for j in v:
            search_seq = ''.join(j)
            key = f'{search_seq}_count'
            res = bio_sequences.apply(
                lambda sequence: sequence.count_overlap(search_seq) / (len(sequence) - len(j) + 1))
            d[key] = res

    return pd.DataFrame(d)


def z_curve(sequences: 'pd.Series[str]') -> pd.DataFrame:
    ### Z-Curve ### total = 3

    if USE_SELECTED_FEATURES["selective"]:
        if 'z_curve_x' in selected_features or 'z_curve_y' in selected_features or 'z_curve_z' in selected_features:
            T = sequences.str.count('T')
            A = sequences.str.count('A')
            C = sequences.str.count('C')
            G = sequences.str.count('G')

            d = {}
            if 'z_curve_x' in selected_features:
                x_ = (A + G) - (C + T)
                d['z_curve_x'] = x_

            if 'z_curve_y' in selected_features:
                y_ = (A + C) - (G + T)
                d['z_curve_y'] = y_

            if 'z_curve_y' in selected_features:
                z_ = (A + T) - (C + G)
                d['z_curve_z'] = z_

            return pd.DataFrame(d)

        return pd.DataFrame()

    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    x_ = (A + G) - (C + T)
    y_ = (A + C) - (G + T)
    z_ = (A + T) - (C + G)

    return pd.DataFrame({'z_curve_x': x_, 'z_curve_y': y_, 'z_curve_z': z_})


def gc_content(sequences: 'pd.Series[str]') -> 'pd.Series[float]':
    if USE_SELECTED_FEATURES["selective"]:
        if 'GC content' not in selected_features:
            return pd.DataFrame()

        T = sequences.str.count('T')
        A = sequences.str.count('A')
        C = sequences.str.count('C')
        G = sequences.str.count('G')

        gc_content = (G + C) / (A + C + G + T)
        return pd.DataFrame({'GC content': gc_content})

    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    gc_content = (G + C) / (A + C + G + T)
    return pd.DataFrame({'GC content': gc_content})


def cumulative_skew(sequences: 'pd.Series[str]') -> pd.DataFrame:
    if USE_SELECTED_FEATURES["selective"]:
        if 'gc_skew' in selected_features or 'at_skew' in selected_features:
            T = sequences.str.count('T')
            A = sequences.str.count('A')
            C = sequences.str.count('C')
            G = sequences.str.count('G')

            d = {}
            if 'gc_skew' in selected_features:
                GCSkew = (G - C) / (G + C)
                d['gc_skew'] = GCSkew
            if 'at_skew' in selected_features:
                ATSkew = (A - T) / (A + T)
                d['at_skew'] = ATSkew
            return pd.DataFrame(d)

        return pd.DataFrame()

    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    GCSkew = (G - C) / (G + C)
    ATSkew = (A - T) / (A + T)

    return pd.DataFrame({'gc_skew': GCSkew, 'at_skew': ATSkew})


def atgc_ratio(sequences: 'pd.Series[str]') -> 'pd.Series[float]':
    if USE_SELECTED_FEATURES["selective"]:
        if 'at/gc_ratio' in selected_features:
            T = sequences.str.count('T')
            A = sequences.str.count('A')
            C = sequences.str.count('C')
            G = sequences.str.count('G')

            atgc_ratio = (A + T) / (G + C)
            return pd.DataFrame({'at/gc_ratio': atgc_ratio})
        return pd.DataFrame()

    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    atgc_ratio = (A + T) / (G + C)
    return pd.DataFrame({'at/gc_ratio': atgc_ratio})


def get_k_gap_description(nucleotides: Tuple[str], before_gap: int, after_gap: int, k: int, gap: str='_') -> str:
    return f'{"".join(nucleotides[:before_gap])}{k*gap}{"".join(nucleotides[before_gap:before_gap+after_gap])}_count'


def mono_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 1___1
    ### g-gap
    '''
    A_A     1-gap
    A__A    2-gap
    A___A   3-gap
    A____A  4-gap
    '''

    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[-1] == _gGap[1]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m2
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 2]

            for gGap in m:
                key = get_k_gap_description(gGap, 1, 1, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m2
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 2]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 1, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def mono_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 1___2
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[-2] == _gGap[1] and v[-1] == _gGap[2]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m3
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 3]

            for gGap in m:
                key = get_k_gap_description(gGap, 1, 2, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))
        return pd.DataFrame(d)

    d = {}
    m = m3
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 3]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 2, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 2___1
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[-1] == _gGap[2]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m3
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 3]

            for gGap in m:
                key = get_k_gap_description(gGap, 2, 1, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m3
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 3]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 1, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def mono_tri_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 1___3
    # A_AAA       1-gap
    # A__AAA      2-gap
    # A___AAA     3-gap
    # A____AAA    4-gap
    # A_____AAA   5-gap upto g
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[-3] == _gGap[1] and v[-2] == _gGap[2] and v[-1] == _gGap[3]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m4
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 4]

            for gGap in m:
                key = get_k_gap_description(gGap, 1, 3, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 3, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def tri_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 3___1
    # AAA_A       1-gap
    # AAA__A      2-gap
    # AAA___A     3-gap
    # AAA____A    4-gap
    # AAA_____A   5-gap upto g
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[2] == _gGap[2] and v[-1] == _gGap[3]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m4
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 4]

            for gGap in m:
                key = get_k_gap_description(gGap, 3, 1, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 3, 1, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 2___2
    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AA       1-gap
    # AA__AA      2-gap
    # AA___AA     3-gap
    # AA____AA    4-gap
    # AA_____AA   5-gap upto g
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[-2] == _gGap[2] and v[-1] == _gGap[3]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m4
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 4]

            for gGap in m:
                key = get_k_gap_description(gGap, 2, 2, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 2, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_tri_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 2___3
    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AAA       1-gap
    # AA__AAA      2-gap
    # AA___AAA     3-gap
    # AA____AAA    4-gap
    # AA_____AAA   5-gap upto g
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[-3] == _gGap[2] and \
                v[-2] == _gGap[3] and v[-1] == _gGap[4]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m5
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 5]

            for gGap in m:
                key = get_k_gap_description(gGap, 2, 3, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m5
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 5]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 3, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def tri_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int) -> pd.DataFrame:  # 3___2
    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AAA_AA       1-gap
    # AAA__AA      2-gap
    # AAA___AA     3-gap
    # AAA____AA    4-gap
    # AAA_____AA   5-gap upto g
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[2] == _gGap[2] and \
                v[-2] == _gGap[3] and v[-1] == _gGap[4]:
                _count += 1
        return _count

    if USE_SELECTED_FEATURES["selective"]:
        d = {}
        m = m5
        for i in tqdm(range(1, g + 1)):
            V = _kmers[i + 5]

            for gGap in m:
                key = get_k_gap_description(gGap, 3, 2, i)
                if key in selected_features:
                    d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

        return pd.DataFrame(d)

    d = {}
    m = m5
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 5]

        for gGap in m:
            key = get_k_gap_description(gGap, 3, 2, i)
            d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def extract_nucli_features(sequences: 'pd.Series[str]') -> pd.DataFrame:
    global selected_features
    selected_features = get_selected_features() if USE_SELECTED_FEATURES["selective"] else None

    d = []
    KMERS = [sequences.apply(lambda sequence: kmers(sequence, i)) for i in range(5 + k_gap + 1)]

    print(f'start z_curve, time: {datetime.now()}')
    res = z_curve(sequences)
    d.append(res)

    print(f'start gc_content, time: {datetime.now()}')
    res = gc_content(sequences)
    d.append(res)

    print(f'start cumulative_skew, time: {datetime.now()}')
    res = cumulative_skew(sequences)
    d.append(res)

    print(f'start atgc_ratio, time: {datetime.now()}')
    res = atgc_ratio(sequences)
    d.append(res)

    print(f'start pseudo_knc, time: {datetime.now()}')
    res = pseudo_knc(sequences, k_tuple)  # k=2|(16), k=3|(64), k=4|(256), k=5|(1024)
    d.append(res)

    print(f'start mono_mono_k_gap, time: {datetime.now()}')
    res = mono_mono_k_gap(KMERS, k_gap)  # 4*(k)*4 = 32
    d.append(res)

    print(f'start mono_di_k_gap, time: {datetime.now()}')
    res = mono_di_k_gap(KMERS, k_gap)  # 4*k*(4^2) = 128
    d.append(res)

    print(f'start mono_tri_k_gap, time: {datetime.now()}')
    res = mono_tri_k_gap(KMERS, k_gap)  # 4*k*(4^3) = 512
    d.append(res)

    print(f'start di_mono_k_gap, time: {datetime.now()}')
    res = di_mono_k_gap(KMERS, k_gap)  # (4^2)*k*(4)    = 128
    d.append(res)

    print(f'start di_di_k_gap, time: {datetime.now()}')
    res = di_di_k_gap(KMERS, k_gap)  # (4^2)*k*(4^2)  = 512
    d.append(res)

    print(f'start di_tri_k_gap, time: {datetime.now()}')
    res = di_tri_k_gap(KMERS, k_gap)  # (4^2)*k*(4^3)  = 2048
    d.append(res)

    print(f'start tri_mono_k_gap, time: {datetime.now()}')
    res = tri_mono_k_gap(KMERS, k_gap)  # (4^3)*k*(4)    = 512
    d.append(res)

    print(f'start tri_di_k_gap, time: {datetime.now()}')
    res = tri_di_k_gap(KMERS, k_gap)  # (4^3)*k*(4^2)  = 2048
    d.append(res)

    return pd.concat(d, axis=1)  # in total with k=2 -> 5943


def get_prob_dict_for_idx(idx_bases):
    counts = {}
    for char in idx_bases:
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    prob_d = {c: counts[c] / len(idx_bases) for c in counts}
    return prob_d


def get_prob_dict(seq_lst: list):
    d = {}
    seq_len = len(seq_lst[0])
    for i in range(seq_len):
        bases_per_idx = [seq[i] for seq in seq_lst]
        d[i] = get_prob_dict_for_idx(bases_per_idx)
    return d


# Features from Tamir's article
# https://www.nature.com/articles/s41598-021-89918-6#Sec8
# See Methods section
def entropy(seq: 'pd.Series[str]') -> pd.DataFrame:
    # calc entropy for seq
    def seq_entropy(seq, prob_dict):
        entropy_val = 0
        for i, v in enumerate(seq):
            p_i = prob_dict[i][v]
            entropy_val += p_i * math.log2(p_i)
        return -entropy_val / 2  # divide by 2 for normalization purposes

    prob_dict = get_prob_dict(seq.to_list())
    # df['entropy'] = df[seq_col].apply(seq_entropy, prob_dict=prob_dict)
    # return df
    return pd.DataFrame(seq.apply(seq_entropy, prob_dict=prob_dict).tolist(), columns=['entropy'])


def run_RNAfold(rna_seq: str):
    # Todo: You need to download RNAfold before from:
    #  https://www.tbi.univie.ac.at/RNA/#download
    #  Include it as part of the project later
    cmd = ['RNAfold', '-p']
    result = subprocess.run(cmd, input=rna_seq, capture_output=True, text=True)
    output = result.stdout.strip().split('\n')
    return output


def get_mfe(rna_seq):
    # Todo: It seems irrelevant in our case
    #  check with the team what they think,
    #  In general folding energy is calculated for RNA,
    #  it supposed to predict minimum free-energy secondary structure of RNA sequence
    #  and in our case the data is only the promotor which is not transcribed to RNA
    """
    Get minimal folding energy for rna seq
    :param rna_seq:
    :return:
    """
    res = run_RNAfold(rna_seq)
    min_fold_energy = float(res[1].split(' (')[1].strip()[:-1])
    return min_fold_energy


def mfe_per_position(rna_seq: str, window_size: int = 31):
    first_half_window = window_size // 2
    second_half_window = window_size // 2 + 1 if window_size % 2 == 1 else window_size // 2
    idx_to_mfe = {}
    for i in range(len(rna_seq)):
        if len(rna_seq) - i >= second_half_window:
            left_i = max(i - first_half_window, 0)
            right_i = left_i + window_size
        else:
            right_i = len(rna_seq)
            left_i = right_i - window_size
        seq = rna_seq[left_i:right_i]
        idx_to_mfe[i] = get_mfe(seq)
        print(f'seq: {seq}, mfe: {get_mfe(seq)}')
    return idx_to_mfe


def add_dict_vals(base_dict, new_dict):
    if base_dict == {}:
        base_dict.update(new_dict)
    else:
        for k, v in base_dict.items():
            base_dict[k] += new_dict[k]


def get_avg_mfe_per_position(df: pd.DataFrame, seq_col: str):
    """
    This function supposed to give what described in the article under Folding energy
    :param df:
    :param seq_col:
    :return:
    """
    avg_mfe_per_position = {}
    for i, r in df.iterrows():
        curr_idx_to_mfe = mfe_per_position(r[seq_col])
        add_dict_vals(avg_mfe_per_position, curr_idx_to_mfe)

    for k, v in avg_mfe_per_position.items():
        avg_mfe_per_position[k] = v / len(df)

    return avg_mfe_per_position


def codon_adaptation_index(gene_seq: str, codon_to_weight: dict):
    # Todo: CAI - another feature that seems irrelevant in our case
    #  It measures the degree with which genes use preferred codons
    #  So it is related to a gene and its codons (and we work with promotor data)
    weights_product = 1
    for i in range(0, len(gene_seq) - 2, 3):
        curr_codon = gene_seq[i:i + 3]
        weights_product *= codon_to_weight[curr_codon]

    num_codons = len(gene_seq) / 3
    return weights_product ** (1 / num_codons)


def effective_codons_number():
    # Todo: ENC measures the degree in which genes use more specific codons
    #  as opposed to using all codons uniformly
    #  So also irrelevant
    pass


def add_mut_in_pos(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def mut_in_pos(seq, pos, base_seq):
        return seq[pos] != base_seq[pos]

    for i in range(len(wildtype_seq)):
        df[f'mut_pos_{i}'] = df[seq_col].apply(mut_in_pos, pos=i, base_seq=wildtype_seq)


def get_short_seq(seq, pos, window_size=31):
    first_half_window = window_size // 2
    second_half_window = window_size // 2 + 1 if window_size % 2 == 1 else window_size // 2
    if len(seq) - pos >= second_half_window:
        left_i = max(pos - first_half_window, 0)
        right_i = left_i + window_size
    else:
        right_i = len(seq)
        left_i = right_i - window_size
    short_seq = seq[left_i:right_i]
    return short_seq


def add_rna_mfe_diff(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def mfe_diff_per_pos(seq, pos, base_seq, window_size=31):
        short_seq = get_short_seq(seq, pos, window_size)
        base_short_seq = get_short_seq(base_seq, pos, window_size)
        ss_mfe = get_mfe(short_seq)
        bss_mfe = get_mfe(base_short_seq)
        return ss_mfe - bss_mfe

    for i in range(len(wildtype_seq)):
        df[f'rna_fe_diff_{i}'] = df[seq_col].apply(mfe_diff_per_pos, pos=i, base_seq=wildtype_seq)


def dna_folding_energy_diff(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    # Todo: It's just a template
    #  need to check how to calc properly MATLAB oligoprop func used in the article
    #  it relates to 4 other resources
    def gibbs_fe_diff_per_pos(seq, pos, base_seq, window_size=31):
        short_seq = get_short_seq(seq, pos, window_size)
        base_short_seq = get_short_seq(base_seq, pos, window_size)
        ss_gfe = MeltingTemp.Tm_NN(short_seq, nn_table=MeltingTemp.R_DNA_NN1)
        bss_gfe = MeltingTemp.Tm_NN(base_short_seq, nn_table=MeltingTemp.R_DNA_NN1)
        return ss_gfe - bss_gfe

    for i in range(len(wildtype_seq)):
        df[f'dna_fe_diff_{i}'] = df[seq_col].apply(gibbs_fe_diff_per_pos, pos=i, base_seq=wildtype_seq)


def run_RNApdist(base_seq: str, seq: str):
    # Todo 1): You need to download RNApdist before from:
    #  https://www.tbi.univie.ac.at/RNA/#download
    #  Include it as part of the project later

    # Todo 2): Verify this is the correct way to run it and results look legit
    input_seq = f'{base_seq}\n{seq}'
    result = subprocess.run(['RNApdist', '-Xf'], input=input_seq, text=True, capture_output=True)
    return result.stdout.strip()


def get_topo(base_rna_seq, rna_seq):
    # Todo: It seems irrelevant in our case
    #  check with the team what they think
    """
    calculates distances between thermodynamic RNA secondary structures ensembles
    Look at 'mRNA topological distance' in the article
    :param rna_seq:
    :return:
    """
    res = run_RNApdist(base_rna_seq, rna_seq)
    topo = float(res)
    return topo


def rna_topo_dist(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def topo_dist_per_pos(seq, pos, base_seq, window_size=31):
        short_seq = get_short_seq(seq, pos, window_size)
        base_short_seq = get_short_seq(base_seq, pos, window_size)
        return get_topo(base_short_seq, short_seq)

    for i in range(len(wildtype_seq)):
        df[f'rna_topo_dist_{i}'] = df[seq_col].apply(topo_dist_per_pos, pos=i, base_seq=wildtype_seq)


def dna_topology_dist_diff(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def curved_dna_diff(seq, base_seq):
        base_seq_params = CurvedDNA(base_seq, model='trifonov')
        seq_params = CurvedDNA(seq, model='trifonov')

        # Todo: return here relevant diff params

    df[['curvature', 'bend_angel', 'curvature_angel', 'helix_x', 'helix_y', 'helix_z',
        'phos_1_x', 'phos_1_y', 'phos_1_z', 'phos_2_x', 'phos_2_y', 'phos_2_z',
        'basepair_n_x', 'basepair_n_y', 'basepair_n_z',
        'smooth_n_x', 'smooth_n_y', 'smooth_n_z']] = df[seq_col].apply(curved_dna_diff, base_seq=wildtype_seq,
                                                                       result_type='expand')

if __name__ == '__main__':
    seq = pd.Series(["TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT",
                     "TTGAAATCCTTTTTTTCTGCGCGTAATCTTTTGCTT",
                     "TAGCGATCCTTTTTTTCTGCCGGTAATCTGCTGCTT",
                     "GTTAGATCCTTTTTTTCTGCGCGTTATACACTGCTT",
                     "TTAGAATCGCCTTTTTCTGCGCGTAATCTGCTAAAT"])
    res = generate_one_hot_encoding(seq)
    # res_df = generate_df_from_seq(seq)
    # res_df2 = extract_nucli_features(seq)
    # res_df.to_csv("origin_version.csv")
    print("done")
