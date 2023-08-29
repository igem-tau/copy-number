from Bio.SeqUtils import MeltingTemp
from dnacurve import CurvedDNA
import pandas as pd
from src.consts import *
import subprocess
from tqdm import tqdm


def extract_base_pairs(structure_line: str) -> dict:
    base_pairs = []
    stack = []
    for idx, char in enumerate(structure_line):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if stack:
                i = stack.pop()
                j = idx
                base_pair = (i, j)
                base_pairs.append(base_pair)
    # Sort the base pairs by the first index
    sorted_base_pairs = sorted(base_pairs, key=lambda pair: pair[0])
    return dict(sorted_base_pairs)


def run_RNAfold_as_webtool(rna_seq: str, params: list = ['RNAfold', '-p', '-d2', '--noLP']):
    result = subprocess.run(params, input=rna_seq, capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def get_rna_secondry_structure(rna_seq: str):
    output = run_RNAfold_as_webtool(rna_seq)
    lines = output.split('\n')
    # sequence_line = lines[1]
    structure_line = lines[2]
    return structure_line


# def get_alpha_beta_match(rna_seq: str):
#     sec_struct = get_rna_secondry_structure(rna_seq)
#     base_pairs_dict = extract_base_pairs(sec_struct)
#     for i in ALPHA_RANGE:
#         base_pairs_dict.get(i)


def get_dist_from_orig_alpha_beta(rna_seq: str):
    sec_struct = get_rna_secondry_structure(rna_seq)
    base_pairs_dict = extract_base_pairs(sec_struct)
    cnt = 0
    for i, j in CONSENSUS_POSITIONS.items():
        if base_pairs_dict.get(i, '-') == j:
            cnt += 1
    return cnt / len(CONSENSUS_POSITIONS)


def get_alpha_area_match_ratio(rna_seq: str, alpha_range: range = EXTENDED_ALPHA_RANGE):
    sec_struct = get_rna_secondry_structure(rna_seq)
    base_pairs_dict = extract_base_pairs(sec_struct)
    hits = 0

    for i in alpha_range:
        if i in base_pairs_dict:
            hits += 1

    return hits / len(alpha_range)


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
        # print(f'seq: {seq}, mfe: {get_mfe(seq)}')
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
    for i, r in tqdm(df.iterrows()):
        curr_idx_to_mfe = mfe_per_position(r[seq_col])
        add_dict_vals(avg_mfe_per_position, curr_idx_to_mfe)

    for k, v in tqdm(avg_mfe_per_position.items()):
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


# no fragmentation version
def add_mut_in_pos(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def mut_in_pos(seq, pos, base_seq):
        return seq[pos] != base_seq[pos]

    mutation_columns = [
        [mut_in_pos(seq, pos=i, base_seq=wildtype_seq) for seq in df[seq_col]]
        for i in range(len(wildtype_seq))
    ]

    new_columns = pd.DataFrame(mutation_columns).transpose()
    new_columns.columns = [f'mut_pos_{i}' for i in range(len(wildtype_seq))]
    new_df = pd.concat([df, new_columns], axis=1)

    return new_df


"""
def add_mut_in_pos(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def mut_in_pos(seq, pos, base_seq):
        return seq[pos] != base_seq[pos]

    for i in range(len(wildtype_seq)):
        df[f'mut_pos_{i}'] = df[seq_col].apply(mut_in_pos, pos=i, base_seq=wildtype_seq)   
"""


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

    mfe_diff_columns = [
        [mfe_diff_per_pos(seq, pos=i, base_seq=wildtype_seq) for seq in df[seq_col]]
        for i in range(len(wildtype_seq))
    ]

    new_columns = pd.DataFrame(mfe_diff_columns, columns=[f'rna_fe_diff_{i}' for i in range(len(wildtype_seq))])
    new_df = pd.concat([df, new_columns], axis=1)

    return new_df

"""
def add_rna_mfe_diff(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def mfe_diff_per_pos(seq, pos, base_seq, window_size=31):
        short_seq = get_short_seq(seq, pos, window_size)
        base_short_seq = get_short_seq(base_seq, pos, window_size)
        ss_mfe = get_mfe(short_seq)
        bss_mfe = get_mfe(base_short_seq)
        return ss_mfe - bss_mfe

    for i in range(len(wildtype_seq)):
        df[f'rna_fe_diff_{i}'] = df[seq_col].apply(mfe_diff_per_pos, pos=i, base_seq=wildtype_seq)
"""



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

    topo_dist_columns = [
        [topo_dist_per_pos(seq, pos=i, base_seq=wildtype_seq) for seq in df[seq_col]]
        for i in range(len(wildtype_seq))
    ]

    new_columns = pd.DataFrame(topo_dist_columns, columns=[f'rna_topo_dist_{i}' for i in range(len(wildtype_seq))])
    df = pd.concat([df, new_columns], axis=1)

    return df


"""
def rna_topo_dist(df: pd.DataFrame, seq_col: str, wildtype_seq: str):
    def topo_dist_per_pos(seq, pos, base_seq, window_size=31):
        short_seq = get_short_seq(seq, pos, window_size)
        base_short_seq = get_short_seq(base_seq, pos, window_size)
        return get_topo(base_short_seq, short_seq)

    for i in range(len(wildtype_seq)):
        df[f'rna_topo_dist_{i}'] = df[seq_col].apply(topo_dist_per_pos, pos=i, base_seq=wildtype_seq)
"""


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


def make_rna_features(rna: pd.DataFrame) -> pd.DataFrame:
    # df["entropy"] = entropy(df[SEQ_COLUMN_NAME])
    pass


def checks():
    res = extract_base_pairs("((((((((....(((((((((...)))))))))))))))))((((..(((((((...)))))))..))))...(((.(((((.((.(((...))).)).))))).)))...")
    tr = run_RNAfold_as_webtool("AGGTGTGTGAACCCGCGCGCGCGCG")
    tr = run_RNAfold("AGGTGTGTGAACCCGCGCGCGCGCG")


if __name__ == '__main__':
    checks()


# features to extract from RNAfold files and RNAeval output

# 1) take rna_p[:111] and make dictionary of the indices:
# Interior loop ( 74,108) CG; ( 75,107) AU:  -210
# Interior loop ( 75,107) AU; ( 76,106) UA:  -110
# Interior loop ( 76,106) UA; ( 78,104) GU:   190
# Interior loop ( 78,104) GU; ( 79,103) AU:  -130
# Interior loop ( 79,103) AU; ( 80,102) CG:  -220
# Interior loop ( 80,102) CG; ( 81,101) CG:  -330
# Interior loop ( 81,101) CG; ( 82,100) GU:  -140
# Interior loop ( 82,100) GU; ( 84, 98) AU:   190
# Interior loop ( 84, 98) AU; ( 85, 97) GC:  -210
# Interior loop ( 85, 97) GC; ( 87, 95) CG:   -60
# Interior loop ( 87, 95) CG; ( 88, 94) GC:  -240
# Interior loop ( 88, 94) GC; ( 89, 93) UG:  -250
# Hairpin  loop ( 89, 93) UG              :   590

# for each sequence take:
# seq[:118], seq[:118+10], etc until seq[:250] beta finish + 53
# for each partial seq check number of matches to the consensus

# same for checking alpha_beta

# using RNAfold output file to get probabilities
# for each partial seq make a feature of length and pair probability,
# for example: seq_120_3_56

# run RNAfold on entire seq and take using RNAeval energy sum of extended alpha beta loop (SL 4)

# run RNAfold on entire seq and look at the C rich area, compare to consensus, take probabilities and energies.