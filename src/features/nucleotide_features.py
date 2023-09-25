from Bio.Seq import Seq
from datetime import datetime
from itertools import product
import math
import pandas as pd
from src.consts import *
from src.utils import is_feature_selected
from typing import List, Optional, Tuple
from tqdm import tqdm


def generate_one_hot_encoding(seq: pd.Series, selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    def encode(single_nucleotid: pd.Series, index: int) -> pd.DataFrame:
        columns = {}
        for nucleotide in NUCLEOTIDES:
            column_name = f'{nucleotide}_{index}'
            if is_feature_selected(column_name, selected_features):
                columns[column_name] = (single_nucleotid == nucleotide).astype(int)

        return pd.DataFrame(columns)

    full_encoding = []
    for current_nucleotide_index in tqdm(range(PROMOTER_LENGTH), desc='one hot encoding'):
        current_nucleotide_encoding = encode(seq.str[current_nucleotide_index], current_nucleotide_index + START_INDEX)
        full_encoding.append(current_nucleotide_encoding)

    return pd.concat(full_encoding, axis=1)


m2 = list(product(NUCLEOTIDES, repeat=2))
m3 = list(product(NUCLEOTIDES, repeat=3))
m4 = list(product(NUCLEOTIDES, repeat=4))
m5 = list(product(NUCLEOTIDES, repeat=5))
k_gap = 2
k_tuple = 2


def kmers(seq: str, k: int) -> List[str]:
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


def pseudo_knc(sequences: 'pd.Series[str]', k: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    ### k-mer ###
    ### A, AA, AAA

    d = {}
    bio_sequences = sequences.apply(Seq)

    for i in tqdm(range(1, k + 1)):
        v = list(product(NUCLEOTIDES, repeat=i))
        for j in v:
            search_seq = ''.join(j)
            key = f'{search_seq}_count'
            if is_feature_selected(key, selected_features):
                res = bio_sequences.apply(lambda sequence: sequence.count_overlap(search_seq) / (len(sequence) - len(j) + 1))
                d[key] = res

    return pd.DataFrame(d)


def z_curve(sequences: 'pd.Series[str]', selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    ### Z-Curve ### total = 3

    if (is_feature_selected('z_curve_x', selected_features) or is_feature_selected('z_curve_y', selected_features) or is_feature_selected('z_curve_z', selected_features)):
        T = sequences.str.count('T')
        A = sequences.str.count('A')
        C = sequences.str.count('C')
        G = sequences.str.count('G')

        d = {}
        if is_feature_selected('z_curve_x', selected_features):
            x_ = (A + G) - (C + T)
            d['z_curve_x'] = x_

        if is_feature_selected('z_curve_y', selected_features):
            y_ = (A + C) - (G + T)
            d['z_curve_y'] = y_

        if is_feature_selected('z_curve_z', selected_features):
            z_ = (A + T) - (C + G)
            d['z_curve_z'] = z_

        return pd.DataFrame(d)


def gc_content(sequences: 'pd.Series[str]') -> pd.DataFrame:
    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    _gc_content = (G + C) / (A + C + G + T)
    return pd.DataFrame({'GC content': _gc_content})


def cumulative_skew(sequences: 'pd.Series[str]', selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    d = {}

    if is_feature_selected('at_skew', selected_features):
        T = sequences.str.count('T')
        A = sequences.str.count('A')
        ATSkew = (A - T) / (A + T)
        d['at_skew'] = ATSkew
    if is_feature_selected('gc_skew', selected_features):
        C = sequences.str.count('C')
        G = sequences.str.count('G')
        GCSkew = (G - C) / (G + C)
        d['gc_skew'] = GCSkew

    return pd.DataFrame()


def atgc_ratio(sequences: 'pd.Series[str]') -> 'pd.Series[float]':
    T = sequences.str.count('T')
    A = sequences.str.count('A')
    C = sequences.str.count('C')
    G = sequences.str.count('G')

    atgc_ratio = (A + T) / (G + C)
    return pd.DataFrame({'at/gc_ratio': atgc_ratio})


def get_k_gap_description(nucleotides: Tuple[str], before_gap: int, after_gap: int, k: int, gap: str='_') -> str:
    return f'{"".join(nucleotides[:before_gap])}{k*gap}{"".join(nucleotides[before_gap:before_gap+after_gap])}_count'


def mono_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 1___1
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

    d = {}
    m = m2
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 2]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 1, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def mono_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 1___2
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[-2] == _gGap[1] and v[-1] == _gGap[2]:
                _count += 1
        return _count

    d = {}
    m = m3
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 3]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 2, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 2___1
    def count_matches(V, _gGap):
        _count = 0
        for v in V:
            if v[0] == _gGap[0] and v[1] == _gGap[1] and v[-1] == _gGap[2]:
                _count += 1
        return _count

    d = {}
    m = m3
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 3]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 1, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def mono_tri_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 1___3
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

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 1, 3, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def tri_mono_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 3___1
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

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 3, 1, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 2___2
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

    d = {}
    m = m4
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 4]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 2, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def di_tri_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 2___3
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

    d = {}
    m = m5
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 5]

        for gGap in m:
            key = get_k_gap_description(gGap, 2, 3, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def tri_di_k_gap(_kmers: 'pd.Series[List[str]]', g: int, selected_features: 'Optional[List[str]]') -> pd.DataFrame:  # 3___2
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

    d = {}
    m = m5
    for i in tqdm(range(1, g + 1)):
        V = _kmers[i + 5]

        for gGap in m:
            key = get_k_gap_description(gGap, 3, 2, i)
            if is_feature_selected(key, selected_features):
                d[key] = V.apply(lambda v: count_matches(v, gGap) / len(v))

    return pd.DataFrame(d)


def extract_nucli_features(sequences: 'pd.Series[str]', selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    d = []
    print("Generating KMERS")
    KMERS = [sequences.apply(lambda sequence: kmers(sequence, i)) for i in tqdm(range(5 + k_gap + 1))]

    print(f'start z_curve, time: {datetime.now()}')
    res = z_curve(sequences, selected_features)
    d.append(res)

    if is_feature_selected('GC content', selected_features):
        print(f'start gc_content, time: {datetime.now()}')
        res = gc_content(sequences)
        d.append(res)

    print(f'start cumulative_skew, time: {datetime.now()}')
    res = cumulative_skew(sequences, selected_features)
    d.append(res)

    if is_feature_selected('at/gc_ratio', selected_features):
        print(f'start atgc_ratio, time: {datetime.now()}')
        res = atgc_ratio(sequences)
        d.append(res)

    print(f'start pseudo_knc, time: {datetime.now()}')
    res = pseudo_knc(sequences, k_tuple, selected_features)  # k=2|(16), k=3|(64), k=4|(256), k=5|(1024)
    d.append(res)

    print(f'start mono_mono_k_gap, time: {datetime.now()}')
    res = mono_mono_k_gap(KMERS, k_gap, selected_features)  # 4*(k)*4 = 32
    d.append(res)

    print(f'start mono_di_k_gap, time: {datetime.now()}')
    res = mono_di_k_gap(KMERS, k_gap, selected_features)  # 4*k*(4^2) = 128
    d.append(res)

    print(f'start mono_tri_k_gap, time: {datetime.now()}')
    res = mono_tri_k_gap(KMERS, k_gap, selected_features)  # 4*k*(4^3) = 512
    d.append(res)

    print(f'start di_mono_k_gap, time: {datetime.now()}')
    res = di_mono_k_gap(KMERS, k_gap, selected_features)  # (4^2)*k*(4)    = 128
    d.append(res)

    print(f'start di_di_k_gap, time: {datetime.now()}')
    res = di_di_k_gap(KMERS, k_gap, selected_features)  # (4^2)*k*(4^2)  = 512
    d.append(res)

    print(f'start di_tri_k_gap, time: {datetime.now()}')
    res = di_tri_k_gap(KMERS, k_gap, selected_features)  # (4^2)*k*(4^3)  = 2048
    d.append(res)

    print(f'start tri_mono_k_gap, time: {datetime.now()}')
    res = tri_mono_k_gap(KMERS, k_gap, selected_features)  # (4^3)*k*(4)    = 512
    d.append(res)

    print(f'start tri_di_k_gap, time: {datetime.now()}')
    res = tri_di_k_gap(KMERS, k_gap, selected_features)  # (4^3)*k*(4^2)  = 2048
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
