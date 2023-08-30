from Bio.SeqUtils import MeltingTemp
from dnacurve import CurvedDNA
import pandas as pd
from src.consts import *
import subprocess
from tqdm import tqdm
import re  # for the base-pairing energy function

CONSENSUS_RNAp_SEQ = 'CGUUUGUUUUUUUGGUGGCGAUGGUCGCCACCAAACAAACGGCCUAGUUCUCGAUGGUUGAGAAAAAGGCUUCCAUUGACCGAAGUCGUCUCGCGUC\
UAUGGUUUAUGACAAGAAGAUCACAUCGGCAUCAAUCCGGUGGUGAAGUUCUUGAGACAUCGUGGCGGAUGUAUGGAGCGAGACGAUUAGGACAAUGGUCACCGACGACGGUCACCGCU\
AUUCAGCACAGAAUGGCCCAACCUGAGUUCUGCUAUCAAUGGCCUAUUCCGCGUCGCCAGCCCGACUUGCCCCCCAAGCACGUGUGUCGGGUCGAACCUCGCUUGCUGGAUGUGGCUUG\
ACUCUAUGGAUGUCGCACUCGAUACUCUUUCGCGGUGCGAAGGGCUUCCCUCUUUCCGCCUGUCCAUAGGCCAUUCGCCGUCCCAGCCUUGUCCUCUCGCGUGCUCCCUCGAAGGUCCC\
CCUUUGCGGACCAUAGAAAUAUCAGGACAGCCCAAAGCGGUGGAGACUGAACUCGCAGCUAAAAACACUACGAGCAGUCCCCCCGCCUCGGAUACCUUUUUGCGGUCGUUGCGCCGGAA'


# convert DNA sequence to the RNA sequence that will be transcripted
def dna_to_rna_complement(dna_sequence):
    complement = {"A": "U", "T": "A", "C": "G", "G": "C"}
    rna_sequence = ""

    for base in dna_sequence:
        if base in complement:
            rna_sequence += complement[base]
        else:
            rna_sequence += base

    return rna_sequence


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


# function to extract base-pairing probabilities of the RNA sequence
# reads the "dot.ps" file that is created when running: run_RNAfold_as_webtool(rna_seq)
# TODO: change the output to fit to the DataFrame, for now the output is a dictionary
def base_pair_probabilities():
    dot_file = r".\dot.ps"
    start_marker = "%start of base pair probability data"
    end_marker = "lbox"  # after all the probabilities, next to appear are the most probable base-pairs
    # which all are assigned to value of 0.95 and the line ends with "lbox" instead of "ubox"

    probabilities_dict = {}
    inside_target_section = False

    with open(dot_file, 'r') as file:
        for line in file:
            line = line.strip()

            if inside_target_section:
                if end_marker in line:
                    inside_target_section = False
                    break
                parts = line.split()  # each line that we read is in this format: "1st_index 2nd_index probability ubox"
                # (for example: 2 14 0.022664034 ubox)
                key = f"{parts[0]} {parts[1]}"  # name the feature "1st_index 2nd_index"
                value = parts[2]  # and assign to it the value 'probability'
                probabilities_dict[key] = value

            if line == start_marker:
                inside_target_section = True

        return probabilities_dict

# assuming that RNAeval also creates a file that seems like the output in the website
# need to choose the Motif to compare, i.e "loop III", "loop IV", "loop VI"
# TODO: change the output to match the DataFrame of the features, now it only returns the energy for the desired loop
def sum_base_pair_energy(loop: str):
    # Define the file path
    file_path = ".\dot.ps"  # Replace with the actual file name

    ranges = {'III': (74, 108), 'IV': (73, 195), 'VI': (216, 328)}
    # Define the range of numbers to look for
    relevant_range = ranges[loop]

    # Initialize the sum
    total_sum = 0

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Extract all numbers using regular expression
            numbers = [int(num) for num in re.findall(r'-?\d+', line)]

            # Check if there are at least two numbers within the specified range
            if relevant_range[0] <= numbers[0] and numbers[1] <= relevant_range[1]:
                print(line)
                # Add the last number to the total sum
                total_sum += numbers[-1]

    return total_sum


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
    # res = extract_base_pairs("((((((((....(((((((((...)))))))))))))))))((((..(((((((...)))))))..))))...(((.(((((.((.(((...))).)).))))).)))...")
    # tr = run_RNAfold_as_webtool("AGGTGTGTGAACCCGCGCGCGCGCG")
    # tr = run_RNAfold("AGGTGTGTGAACCCGCGCGCGCGCG")
    prob_dict = base_pair_probabilities()
    bp_ener = sum_base_pair_energy('VI')


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

# using RNAfold output file to get probabilities DONE:) base_pair_probabilities()

# TODO: for each partial seq make a feature of length and pair probability, for example: "seq_120_3_56"

# run RNAfold on entire seq and take using RNAeval energy sum of extended alpha beta loop (SL 4)
# stem-loop IV seq[72:194]
# Interior loop ( 73,195) CG; ( 74,194) CG:  -330
# Interior loop ( 74,194) CG; ( 75,193) AU:  -210
# Interior loop ( 75,193) AU; ( 76,192) UA:  -110
# Interior loop ( 76,192) UA; ( 77,191) UA:   -90
# Interior loop ( 77,191) UA; ( 78,190) GC:  -210
# Interior loop ( 78,190) GC; ( 80,188) CG:    80
# Interior loop ( 80,188) CG; ( 81,187) CG:  -330
# Interior loop ( 81,187) CG; ( 84,185) AU:   190
# Interior loop ( 84,185) AU; ( 85,184) GU:   -60
# Interior loop ( 85,184) GU; ( 86,183) UA:  -140
# Interior loop ( 86,183) UA; ( 87,182) CG:  -240
# Interior loop ( 87,182) CG; ( 88,181) GC:  -240
# Interior loop ( 88,181) GC; ( 89,180) UA:  -220
# Interior loop ( 89,180) UA; ( 90,179) CG:  -240
# Interior loop ( 90,179) CG; ( 91,178) UA:  -210
# Interior loop ( 91,178) UA; ( 92,177) CG:  -240
# Interior loop ( 92,177) CG; ( 93,176) GC:  -240
# Interior loop ( 93,176) GC; ( 94,175) CG:  -340
# Interior loop ( 94,175) CG; ( 96,174) UA:   170
# Interior loop ( 96,174) UA; ( 97,173) CG:  -240
# Interior loop ( 97,173) CG; ( 98,172) UG:  -210
# Interior loop ( 98,172) UG; ( 99,171) AU:  -100
# Interior loop ( 99,171) AU; (100,170) UA:  -110
# Interior loop (110,151) CG; (111,150) AU:  -210
# Interior loop (111,150) AU; (112,149) AU:   -90
# Interior loop (112,149) AU; (113,148) GC:  -210
# Interior loop (113,148) GC; (114,147) AU:  -240
# Interior loop (114,147) AU; (115,146) AU:   -90
# Interior loop (115,146) AU; (118,143) UA:   100
# Interior loop (118,143) UA; (119,142) CG:  -240
# Interior loop (119,142) CG; (120,141) AU:  -210
# Interior loop (120,141) AU; (121,140) CG:  -220
# Interior loop (121,140) CG; (122,138) AU:   170
# Interior loop (122,138) AU; (123,137) UG:  -140
# Interior loop (123,137) UG; (124,136) CG:  -150
# Interior loop (124,136) CG; (125,135) GC:  -240
# Interior loop (125,135) GC; (126,134) GC:  -330
# Hairpin  loop (126,134) GC              :   550
# Interior loop (154,169) AU; (155,168) CG:  -220
# Interior loop (155,168) CG; (156,167) AU:  -210
# Interior loop (156,167) AU; (157,166) UA:  -110
# Interior loop (157,166) UA; (158,165) CG:  -240
# Hairpin  loop (158,165) CG              :   300
# Multi    loop (100,170) UA              :   460

# run RNAfold on entire seq and look at the C rich area, compare to consensus, take probabilities and energies.
# C rich area seq[285:291]  - should be in the loop and not base-paired
# the whole stem-loop VI  seq[215:327]
# Interior loop (216,328) UG; (217,327) AU:  -100
# Interior loop (217,327) AU; (218,326) UA:  -110
# Interior loop (218,326) UA; (219,325) UG:  -130
# Interior loop (219,325) UG; (220,324) CG:  -150
# Interior loop (220,324) CG; (221,323) AU:  -210
# Interior loop (221,323) AU; (222,322) GC:  -210
# Interior loop (222,322) GC; (223,321) CG:  -340
# Interior loop (223,321) CG; (224,320) AU:  -210
# Interior loop (227,265) GC; (228,264) AU:  -240
# Interior loop (228,264) AU; (229,263) AU:   -90
# Interior loop (229,263) AU; (230,262) UA:  -110
# Interior loop (230,262) UA; (231,260) GC:   170
# Interior loop (231,260) GC; (232,259) GC:  -330
# Interior loop (232,259) GC; (233,258) CG:  -340
# Interior loop (233,258) CG; (234,257) CG:  -330
# Interior loop (234,257) CG; (242,250) AU:   330
# Interior loop (242,250) AU; (243,249) GC:  -210
# Interior loop (243,249) GC; (244,248) UG:  -250
# Hairpin  loop (244,248) UG              :   590
# Interior loop (267,318) GC; (268,317) CG:  -340
# Interior loop (268,317) CG; (269,316) GC:  -240
# Interior loop (269,316) GC; (270,311) UA:   410
# Interior loop (270,311) UA; (271,310) CG:  -240
# Interior loop (271,310) CG; (272,309) GC:  -240
# Interior loop (272,309) GC; (276,308) GU:   370
# Interior loop (276,308) GU; (277,307) CG:  -250
# Interior loop (277,307) CG; (278,306) CG:  -330
# Interior loop (278,306) CG; (279,305) CG:  -330
# Interior loop (279,305) CG; (280,304) GC:  -240
# Interior loop (280,304) GC; (281,303) AU:  -240
# Interior loop (281,303) AU; (282,302) CG:  -220
# Interior loop (282,302) CG; (285,299) GU:   120
# Interior loop (285,299) GU; (286,298) CG:  -250
# Hairpin  loop (286,298) CG              :   550
# Multi    loop (224,320) AU              :   400

