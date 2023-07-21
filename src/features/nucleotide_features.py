from Bio.SeqUtils import MeltingTemp
from dnacurve import CurvedDNA
import itertools
import math
import pandas as pd
from src.consts import *
import subprocess
from typing import List, Dict, Tuple, Union


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

    full_encoding = []
    for current_nucleotide_index in range(PROMOTER_LENGTH):
        current_nucleotide_encoding = encode(seq.str[current_nucleotide_index], current_nucleotide_index + START_INDEX)
        full_encoding.append(current_nucleotide_encoding)

    return pd.concat(full_encoding, axis=1)


m2 = list(itertools.product(NUCLEOTIDES, repeat=2))
m3 = list(itertools.product(NUCLEOTIDES, repeat=3))
m4 = list(itertools.product(NUCLEOTIDES, repeat=4))
m5 = list(itertools.product(NUCLEOTIDES, repeat=5))
k_gap = 2
k_tuple = 2


def kmers(seq: str, k: int) -> List[int]:
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


def pseudo_knc(seq: str, k: int) -> Dict[str, int]:
    ### k-mer ###
    ### A, AA, AAA

    d = {}
    for i in range(1, k + 1):
        v = list(itertools.product(NUCLEOTIDES, repeat=i))
        for j in v:
            search_seq = ''.join(j)
            key = f'{search_seq}_count'
            res = seq.count(search_seq)
            d.update({key: res})
    return d


def z_curve(seq: str) -> Dict[str, int]:
    ### Z-Curve ### total = 3

    T = seq.count('T')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    x_ = (A + G) - (C + T)
    y_ = (A + C) - (G + T)
    z_ = (A + T) - (C + G)

    d = {'z_curve_x_': x_, 'z_curve_y_': y_, 'z_curve_z_': z_}
    return d


def gc_content(seq: str) -> Dict[str, float]:
    T = seq.count('T')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    gc_content = (G + C) / (A + C + G + T)
    return {'gc_content': gc_content}


def cumulative_skew(seq: str) -> Dict[str, float]:
    T = seq.count('T')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    GCSkew = (G - C) / (G + C)
    ATSkew = (A - T) / (A + T)

    d = {'gc_skew': GCSkew, 'at_skew': ATSkew}
    return d


def atgc_ratio(seq: str) -> Dict[str, float]:
    T = seq.count('T')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    atgc_ratio = (A + T) / (G + C)
    return {'atgc_ratio': atgc_ratio}


def get_k_gap_description(nucleotides: Tuple[str], before_gap: int, after_gap: int, k: int, gap: str = '_') -> str:
    return f'{"".join(nucleotides[:before_gap])}{k * gap}{"".join(nucleotides[before_gap:before_gap + after_gap])}_count'


def mono_mono_k_gap(seq: str, g: int) -> Dict[str, int]:  # 1___1
    ### g-gap
    """
    AA      0-gap (2-mer)
    A_A     1-gap
    A__A    2-gap
    A___A   3-gap
    A____A  4-gap
    """

    d = {}
    m = m2
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 2)

        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            key = get_k_gap_description(gGap, 1, 1, i)
            d[key] = C

    return d


def mono_di_k_gap(seq: str, g: int) -> Dict[str, int]:  # 1___2

    d = {}
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 3)
        for gGap in m:

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1

            key = get_k_gap_description(gGap, 1, 2, i)
            d[key] = C

    return d


def di_mono_k_gap(seq: str, g: int) -> Dict[str, int]:  # 2___1

    d = {}
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 3)

        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            key = get_k_gap_description(gGap, 2, 1, i)
            d.update({key: C})

    return d


def mono_tri_k_gap(seq: str, g: int) -> Dict[str, int]:  # 1___3

    # A_AAA       1-gap
    # A__AAA      2-gap
    # A___AAA     3-gap
    # A____AAA    4-gap
    # A_____AAA   5-gap upto g

    d = {}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 4)

        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-3] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            key = get_k_gap_description(gGap, 1, 3, i)
            d[key] = C

    return d


def tri_mono_k_gap(seq: str, g: int) -> Dict[str, int]:  # 3___1

    # AAA_A       1-gap
    # AAA__A      2-gap
    # AAA___A     3-gap
    # AAA____A    4-gap
    # AAA_____A   5-gap upto g

    d = {}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 4)

        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1

            key = get_k_gap_description(gGap, 3, 1, i)
            d[key] = C

    return d


def di_di_k_gap(seq: str, g: int) -> Dict[str, int]:
    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AA       1-gap
    # AA__AA      2-gap
    # AA___AA     3-gap
    # AA____AA    4-gap
    # AA_____AA   5-gap upto g

    d = {}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 4)

        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            key = get_k_gap_description(gGap, 2, 2, i)
            d[key] = C

    return d


def di_tri_k_gap(seq: str, g: int) -> Dict[str, int]:  # 2___3

    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AAA       1-gap
    # AA__AAA      2-gap
    # AA___AAA     3-gap
    # AA____AAA    4-gap
    # AA_____AAA   5-gap upto g

    d = {}
    m = m5
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-3] == gGap[2] and \
                        v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            key = get_k_gap_description(gGap, 2, 3, i)
            d[key] = C

    return d


def tri_di_k_gap(seq: str, g):  # 3___2

    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AAA_AA       1-gap
    # AAA__AA      2-gap
    # AAA___AA     3-gap
    # AAA____AA    4-gap
    # AAA_____AA   5-gap upto g

    d = {}
    m = m5
    for i in range(1, g + 1, 1):
        V = kmers(seq, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and \
                        v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            key = get_k_gap_description(gGap, 3, 2, i)
            d[key] = C

    return d


def extract(seq: str) -> Dict[str, Union[int, float]]:
    d = {}

    res = z_curve(seq)
    d.update(res)

    res = gc_content(seq)
    d.update(res)

    res = cumulative_skew(seq)
    d.update(res)

    res = atgc_ratio(seq)
    d.update(res)

    res = pseudo_knc(seq, k_tuple)  # k=2|(16), k=3|(64), k=4|(256), k=5|(1024)
    d.update(res)

    res = mono_mono_k_gap(seq, k_gap)  # 4*(k)*4 = 240
    d.update(res)

    res = mono_di_k_gap(seq, k_gap)  # 4*k*(4^2) = 960
    d.update(res)

    res = mono_tri_k_gap(seq, k_gap)  # 4*k*(4^3) = 3,840
    d.update(res)

    res = di_mono_k_gap(seq, k_gap)  # (4^2)*k*(4)    = 960
    d.update(res)

    res = di_di_k_gap(seq, k_gap)  # (4^2)*k*(4^2)  = 3,840
    d.update(res)

    res = di_tri_k_gap(seq, k_gap)  # (4^2)*k*(4^3)  = 15,360
    d.update(res)

    res = tri_mono_k_gap(seq, k_gap)  # (4^3)*k*(4)    = 3,840
    d.update(res)

    res = tri_di_k_gap(seq, k_gap)  # (4^3)*k*(4^2)  = 15,360
    d.update(res)

    return d


def generate_df_from_seq(seq: 'pd.Series[str]') -> pd.DataFrame:
    return pd.DataFrame(seq.apply(extract).tolist())


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
