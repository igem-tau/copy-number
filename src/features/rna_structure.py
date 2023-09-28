from Bio.SeqUtils import MeltingTemp
from dnacurve import CurvedDNA
from multiprocessing import Process, Manager
import os
import pandas as pd
from src.consts import *
import subprocess
import re  # for the base-pairing energy function
from tqdm import tqdm
from typing import List
from time import time


CONSENSUS_RNAp_SEQ = 'UGCAAACAAAAAAACCACCGCUACCAGCGGUGGUUUGUUUGCCGGAUCAAGAGCUACCAACUCUUUUUCCGAAGGUAACUGGCUUCAGCAGAGCGCAGAUACCAAAUACUGUUCUUCUAGUGUAGCCGUAGUUAGGCCACCACUUCAAGAACUCUGUAGCACCGCCUACAUACCUCGCUCUGCUAAUCCUGUUACCAGUGGCUGCUGCCAGUGGCGAUAAGUCGUGUCUUACCGGGUUGGACUCAAGACGAUAGUUACCGGAUAAGGCGCAGCGGUCGGGCUGAACGGGGGGUUCGUGCACACAGCCCAGCUUGGAGCGAACGACCUACACCGAACAGAUACCUACAGCGUGAGCUAUGAGAAAGCGCCACGCUUCCCGAAGGGAGAAAGGCGGACAGGUAUCCGGUAAGCGGCAGGGUCGGAACAGGAGAGCGCACGAGGGAGCUUCCAGGGGGAAACGCCUGGUAUCUUUAUAGUCCUGUCGGGUUUCGCCACCUCUGACUUGAGCGUCGAUUUUUGUGAUGCUCGUCAGGGGGGCGGAGCCUAUGGAAAA'

STEM_LOOPS = ["III", "IV"]

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


def run_RNAfold_as_webtool(rna_seq: str, params: list = [r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe", '-p', '-d2', '--noLP', '--noPS', '--noDP']):
    result = subprocess.run(params, input=rna_seq, capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def get_rna_secondry_structure(rna_seq: str):
    output = run_RNAfold_as_webtool(rna_seq)
    lines = output.split('\n')
    mfe_structure_line = lines[1]
    mfe_structure_line = mfe_structure_line.split(" ")[0]
    # todo: consider taking other formats like line 2 or 3
    centroid_structure_line = lines[3]
    centroid_structure_line = centroid_structure_line.split(" ")[0]
    return mfe_structure_line, centroid_structure_line


# def get_alpha_beta_match(rna_seq: str):
#     sec_struct = get_rna_secondry_structure(rna_seq)
#     base_pairs_dict = extract_base_pairs(sec_struct)
#     for i in ALPHA_RANGE:
#         base_pairs_dict.get(i)


def get_dist_from_orig_alpha_beta(rna_seq: str):
    sec_struct, _ = get_rna_secondry_structure(rna_seq)
    base_pairs_dict = extract_base_pairs(sec_struct)
    cnt = 0
    for i, j in CONSENSUS_POSITIONS_ALPHA_BETA_FOLD.items():
        if base_pairs_dict.get(i, '-') == j:
            cnt += 1
    return cnt / len(CONSENSUS_POSITIONS_ALPHA_BETA_FOLD)


def get_alpha_area_match_ratio(rna_seq: str, alpha_range: range = EXTENDED_ALPHA_RANGE):
    sec_struct, _ = get_rna_secondry_structure(rna_seq)
    base_pairs_dict = extract_base_pairs(sec_struct)
    hits = 0

    for i in alpha_range:
        if i in base_pairs_dict:
            hits += 1

    return hits / len(alpha_range)


def get_match_ratio(rna_seq: str, consensus: dict) -> float:
    sec_struct, _ = get_rna_secondry_structure(rna_seq)
    bp_dict = extract_base_pairs(sec_struct)
    hits = 0
    for k, v in consensus.items():
        hits += 1 if bp_dict.get(k, -1) == v else 0
    return hits / len(consensus)


def get_match_rate_to_stem_loop_3(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    print("Running get_match_rate_to_stem_loop_3")
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        col_desc = f"sl3_match_seq_end_{end_idx}"
        d[col_desc] = partial_seqs.apply(lambda seq: get_match_ratio(seq, CONSENSUS_POSITIONS_3_STEM_LOOPS))
    return pd.DataFrame(d)


def get_match_rate_to_alpha_beta(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    print("Running get_match_rate_to_alpha_beta")
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        col_desc = f"alpha_beta_match_seq_end_{end_idx}"
        d[col_desc] = partial_seqs.apply(lambda seq: get_match_ratio(seq, CONSENSUS_POSITIONS_ALPHA_BETA_FOLD))
    return pd.DataFrame(d)


def get_match_rate_to_extended_alpha_beta(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    print("Running get_match_rate_to_extended_alpha_beta")
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        col_desc = f"alpha_beta_extended_match_seq_end_{end_idx}"
        d[col_desc] = partial_seqs.apply(lambda seq: get_match_ratio(seq, CONSENSUS_POSITIONS_EXTENDED_ALPHA_BETA_FOLD))
    return pd.DataFrame(d)


def get_match_rate_to_c_rich_area(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    print("Running get_match_rate_to_c_rich_area")
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        col_desc = f"c_rich_area_match_seq_end_{end_idx}"
        d[col_desc] = partial_seqs.apply(lambda seq: get_match_ratio(seq, CONSENSUS_POSITIONS_C_RICH_AREA))
    return pd.DataFrame(d)


def get_base_pairing_ranges(seqs: 'pd.Series[List[str]]') -> pd.DataFrame:
    d = {}
    for k, r in tqdm(RANGES_DICT.items()):
        col_desc_mfe = f"base_pairing_mfe_{k}"
        d[col_desc_mfe] = seqs.apply(lambda seq: get_range_paired_bases(seq, r, "mfe"))
        col_desc_centroid = f"base_pairing_centorid_{k}"
        d[col_desc_centroid] = seqs.apply(lambda seq: get_range_paired_bases(seq, r, "centroid"))
    return pd.DataFrame(d)


def get_rna_form(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        unpaired_cnt = f"unpaired_cnt_seq_end_{end_idx}"
        bow_cnt = f"bow_cnt_seq_end_{end_idx}"
        bubble_cnt = f"bubble_cnt_seq_end_{end_idx}"
        result = partial_seqs.apply(lambda seq: extract_rna_form(seq, 2, 124))
        d[unpaired_cnt], d[bow_cnt], d[bubble_cnt] = zip(*result)
    res_df = pd.DataFrame(d)
    res_df.index = seqs.index
    return res_df


def rna_features_in_window(rna_seq: str):
    mfe = get_mfe(rna_seq)
    unpaired_cnt, bow_cnt, bubble_cnt = extract_rna_form(rna_seq, 0, len(rna_seq))
    return mfe, unpaired_cnt, bow_cnt, bubble_cnt


def rna_features_by_windows(seqs: 'pd.Series[List[str]]', window_start: int = 0, window_end: int = 60, window_size: int = 70, window_jump: int = 10) -> pd.DataFrame:
    d = {}
    while tqdm(window_start <= window_end):
        partial_seqs = seqs.apply(lambda seq: seq[window_start:window_start + window_size])
        result = partial_seqs.apply(lambda seq: rna_features_in_window(seq))
        mfe = f"mfe_window_{window_start}_{window_start+window_size}"
        unpaired_cnt = f"unpaired_cnt_window_{window_start}_{window_start+window_size}"
        bow_cnt = f"bow_cnt_window_{window_start}_{window_start+window_size}"
        bubble_cnt = f"bubble_cnt_window_{window_start}_{window_start+window_size}"
        d[mfe], d[unpaired_cnt], d[bow_cnt], d[bubble_cnt] = zip(*result)
        window_start += window_jump
    res_df = pd.DataFrame(d)
    res_df.index = seqs.index
    return res_df


# function to extract base-pairing probabilities of the RNA sequence
# reads the "dot.ps" file that is created when running: run_RNAfold_as_webtool(rna_seq)
# TODO: change the output to fit to the DataFrame, for now the output is a dictionary
def base_pair_probabilities(file_path: str, end_idx=None) -> dict:
    # dot_file = r".\dot.ps"
    start_marker = "%start of base pair probability data"
    end_marker = "lbox"  # after all the probabilities, next to appear are the most probable base-pairs
    # which all are assigned to value of 0.95 and the line ends with "lbox" instead of "ubox"
    finish_marker = "showpage" # sometimes there is no lbox and after the lines we have "showpage" line

    probabilities_dict = {}
    inside_target_section = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if inside_target_section:
                if end_marker in line or finish_marker in line:
                    inside_target_section = False
                    break

                parts = line.split()  # each line that we read is in this format: "1st_index 2nd_index probability ubox"
                # (for example: 2 14 0.022664034 ubox)
                try:
                    if end_idx:
                        key = f"seq_end_{end_idx}_{parts[0]}_{parts[1]}"  # name the feature "1st_index 2nd_index"
                    else:
                        key = f"{parts[0]}_{parts[1]}"
                    value = parts[2]  # and assign to it the value 'probability'
                    probabilities_dict[key] = value
                except Exception as ex:
                    print(ex)

            if line == start_marker:
                inside_target_section = True

        return probabilities_dict


def get_prob_for_seq(rna_seq: str, end_idx=None) -> dict:
    run_RNAfold(rna_seq, params=[r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe", '-p', '-d2', '--noLP', '--noPS'])

    # check we got file
    assert os.path.exists("dot.ps"), "dot file with probabilities generation failed"
    res = base_pair_probabilities("dot.ps", end_idx)

    # remove files
    os.remove("dot.ps")
    # os.remove("rna.ps")

    return res


def get_mfe_for_seq(seq: str, stem_loop_type: str):
    output = run_RNAfold_as_webtool(seq)
    secondry_structure = output.split()[1]
    energy_data = run_RNAeval(seq, secondry_structure)
    res = sum_base_pair_energy(energy_data, stem_loop_type)
    return res


def get_stem_loops_mfe(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    d = {}
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        for sl in STEM_LOOPS:
            col_desc = f"mfe_seq_end_{end_idx}_sl_{sl}"
            d[col_desc] = partial_seqs.apply(lambda seq: get_mfe_for_seq(seq, sl))
    return pd.DataFrame(d)


def get_prob_df(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    df_ls = []
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        seq_and_prob_df = partial_seqs.apply(lambda seq: get_prob_for_seq(seq, end_idx))
        df = pd.DataFrame(seq_and_prob_df.tolist())
        df.fillna(0, inplace=True)
        df_ls.append(df)
    res_df = pd.concat(df_ls, axis=1)
    res_df.index = seqs.index
    return res_df


# assuming that RNAeval also creates a file that seems like the output in the website
# need to choose the Motif to compare, i.e "loop III", "loop IV", "loop VI"
# TODO: change the output to match the DataFrame of the features, now it only returns the energy for the desired loop
def sum_base_pair_energy(energy_data: List[str], loop: str) -> int:
    ranges = {'III': (3, 32), 'IV': (3, 125)}
    # Define the range of numbers to look for
    relevant_range = ranges[loop]

    # Initialize the sum
    total_sum = 0

    for line in energy_data:
        # Extract all numbers using regular expression
        numbers = [int(num) for num in re.findall(r'-?\d+', line)]
        # checks if the line of correct expected format
        # for example: 'Interior loop (  3, 20) GC; (  4, 19) UG:  -250'
        if len(numbers) != 5:
            continue

        # Check if there are at least two numbers within the specified range
        if relevant_range[0] <= numbers[0] and numbers[1] <= relevant_range[1]:
            # print(line)
            # Add the last number to the total sum
            total_sum += numbers[-1]

    return total_sum


# gets MFE and Centroid structures, compares them and returns a dictionary of the matching base pairs
def compare_mfe_to_centroid(rna_seq: str, end_idx: int) -> dict:
    mfe_structure, centroid_structure = get_rna_secondry_structure(rna_seq)
    mfe_bp = extract_base_pairs(mfe_structure)
    centroid_bp = extract_base_pairs(centroid_structure)
    identical_bp = {"seq_end_%s mfe == centroid %s_%s" %(end_idx, k, v): 1 for k, v in mfe_bp.items() if k in centroid_bp and centroid_bp[k] == v}
    return identical_bp


def get_mfe_centroid_comparison_df(seqs: 'pd.Series[List[str]]', seq_end_idx: list) -> pd.DataFrame:
    df_ls = []
    for end_idx in tqdm(seq_end_idx):
        partial_seqs = seqs.apply(lambda seq: seq[:end_idx])
        seq_mfe_centroid_comparison_df = partial_seqs.apply(lambda seq: compare_mfe_to_centroid(seq, end_idx))
        df = pd.DataFrame(seq_mfe_centroid_comparison_df.tolist())
        df.fillna(0, inplace=True)
        df_ls.append(df)
    res_df = pd.concat(df_ls, axis=1)
    res_df.index = seqs.index
    return res_df


#how many alpha/beta/gammma bases are paired with any other bases
def get_range_paired_bases(rna_seq: str, stem_loop_range: range = ALPHA_RANGE, fold_type: str = "mfe"):
    mfe_sec_struct, centroid_sec_struct = get_rna_secondry_structure(rna_seq)
    if fold_type == "mfe":
        base_pairs_dict = extract_base_pairs(mfe_sec_struct)
    elif fold_type == "centroid":
        base_pairs_dict = extract_base_pairs(centroid_sec_struct)
    else:
        raise Exception(f"Invalid folding type requested in: get_range_paired_bases")

    hits = 0
    for i in stem_loop_range:
        if i in base_pairs_dict.keys() or i in base_pairs_dict.values():
            hits += 1
    return hits / len(stem_loop_range)


#how many alpha/beta/gammma bases are unpaired
def extract_rna_form(rna_seq: str, start_idx: int, end_idx: int):
    #get unpaired bases
    sec_struct, _ = get_rna_secondry_structure(rna_seq)
    unpaired_bases_idx = []

    for idx, char in enumerate(sec_struct):
        if char == ".":
            unpaired_bases_idx.append(idx)
    # unpaired bases in alpha beta extended
    unpaired_bases_idx_ab = [x for x in unpaired_bases_idx if start_idx <= x <= end_idx]
    num_unpaired_ab = len(unpaired_bases_idx_ab)

    # get bows number
    cnt_bow = 0
    bows = []
    mini_bow = []
    for ind, base_idx in enumerate(unpaired_bases_idx_ab):
        mini_bow.append(base_idx)
        if ind == len(unpaired_bases_idx_ab) -1 or unpaired_bases_idx_ab[ind+1] > unpaired_bases_idx_ab[ind]+1  :
            bows.append(mini_bow)
            cnt_bow += 1
            mini_bow = []

    # get bubbles number, bubble is 1 or more unpaired bases from both sides of 1 or 2 paired bases
    cnt_bub = 0
    bubbles = []
    base_pairs_dict = extract_base_pairs(sec_struct)
    dict_ab = {key: value for key, value in base_pairs_dict.items() if start_idx <= key <= end_idx}
    dict_items = sorted(dict_ab.items())

    # dict_items[ind2][0] key dict_items[ind2][1] value
    for ind1, bow in enumerate(bows):
        for ind2, (key, value) in enumerate(dict_items):
            if bow[0] in [item for sublist in bubbles for item in sublist]:
                continue

            if ind2 == len(dict_items) - 1 and bow[0] == dict_items[ind2][0]+1 and bow[-1] == dict_items[ind2][1]-1:
                cnt_bub += 1
                bubbles.append(bow)

            if not ind2 == len(dict_items) - 1 and bow[0] == dict_items[ind2][0]+1 and bow[-1] == dict_items[ind2][1]-1:
                cnt_bub += 1
                bubbles.append(bow)

            if not ind2 == len(dict_items) - 1 and bow[0] > dict_items[ind2][0] and bow[-1] < dict_items[ind2+1][0]:
                if dict_items[ind2][1]-dict_items[ind2+1][1] > 1: #Values are ordered opposite to the keys
                    size_sec_bow = dict_items[ind2][1]-dict_items[ind2+1][1]
                    ind_sec_bow = list(range(dict_items[ind2+1][1]+1, dict_items[ind2+1][1]+size_sec_bow))

                    if ind_sec_bow in bows:
                        cnt_bub += 1
                        bow_copy = bow[:]
                        bow_copy.append(ind_sec_bow)
                        flat_bow_copy = [item for sublist in bow_copy for item in (sublist if isinstance(sublist, list) else [sublist])]
                        bubbles.append(flat_bow_copy)

    return num_unpaired_ab, cnt_bow, cnt_bub


def run_RNAeval(rna_seq: str, secondry_structure: str):
    # Todo: You need to download RNAfold before from:
    #  https://www.tbi.univie.ac.at/RNA/#download
    #  Include it as part of the project later
    cmd = [r"C:\Program Files (x86)\ViennaRNA Package\RNAeval.exe", '-v']
    result = subprocess.run(cmd, input=f"{rna_seq}\n{secondry_structure}", capture_output=True, text=True)
    output = result.stdout.strip().split('\n')
    return output


def run_RNAfold(rna_seq: str, params: list = [r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe", '-p', '-d2', '--noLP', '--noPS', '--noDP']):
    # Todo: You need to download RNAfold before from:
    #  https://www.tbi.univie.ac.at/RNA/#download
    #  Include it as part of the project later
    # cmd = ['RNAfold', '-p', '-d2', '--noLP']    # how they run it in webtool
    result = subprocess.run(params, input=rna_seq, capture_output=True, text=True)
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
        'smooth_n_x', 'smooth_n_y', 'smooth_n_z']] = df[seq_col].apply(curved_dna_diff, base_seq=wildtype_seq, result_type='expand')


def make_rna_features(rna: pd.DataFrame) -> pd.DataFrame:
    sl_match = get_match_rate_to_stem_loop_3(rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483])
    alpha_beta_match = get_match_rate_to_alpha_beta(rna, [130, 180, 230, 300, 400, 483])
    alpha_beta_extended_match = get_match_rate_to_extended_alpha_beta(rna, [130, 180, 230, 300, 400, 483])
    # c_rich_area_match = get_match_rate_to_c_rich_area(rna, [230, 300, 400, 483])  # there is no consensus stem loop VI
    bases_probabilities = get_prob_df(rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483])  # check only specific "interesting locations"
    mfe_centroid_comparison = get_mfe_centroid_comparison_df(rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483])
    stem_loops_mfe = get_stem_loops_mfe(rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483])
    base_pairing_ranges = get_base_pairing_ranges(rna)
    rna_forms = get_rna_form(rna, range(12, 133, 10))
    features_by_windows = rna_features_by_windows(rna)

    features_dfs = [sl_match, alpha_beta_match, alpha_beta_extended_match, bases_probabilities, mfe_centroid_comparison, stem_loops_mfe, base_pairing_ranges, rna_forms, features_by_windows]
    # excluded c_rich_area_match
    result_df = pd.concat(features_dfs, axis=1)
    return result_df

#
# def make_rna_features_parallel(rna: pd.DataFrame) -> pd.DataFrame:
#     pool = multiprocessing.Pool(processes=2)
#
#     result_a = pool.apply_async(a, (p1, p2))


def worker(func_and_args, results):
    func, args = func_and_args
    result = func(*args)
    results.append(result)


def make_rna_features_parallel(rna: pd.DataFrame) -> pd.DataFrame:
    # Define the list of inner functions and their corresponding arguments
    functions_and_args = [
        (get_match_rate_to_stem_loop_3, [rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483]]),
        (get_match_rate_to_alpha_beta, [rna, [130, 180, 230, 300, 400, 483]]),
        (get_match_rate_to_extended_alpha_beta, [rna, [130, 180, 230, 300, 400, 483]]),
        # (get_match_rate_to_c_rich_area, [rna, [230, 300, 400, 483]]),
        (get_prob_df, [rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483]]),
        (get_mfe_centroid_comparison_df, [rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483]]),
        (get_stem_loops_mfe, [rna, [50, 60, 70, 80, 130, 180, 230, 300, 400, 483]]),
        (get_base_pairing_ranges, [rna]),
        (get_rna_form, [rna, list(range(12, 133, 10))]),
        (rna_features_by_windows, [rna])
    ]

    manager = Manager()
    results = manager.list()

    processes = []

    for func_and_args in functions_and_args:
        process = Process(target=worker, args=(func_and_args, results))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    # Concatenate the results into the final DataFrame
    result_df = pd.concat(results, axis=1)

    return result_df


def generate_features_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_file_path = os.path.join(script_dir, '..', '..', 'data/rna_p_data.csv')
    csv_file_path = r'C:\Users\YH006_new\Desktop\copy-number\data\rna_p_data'
    df = pd.read_csv(csv_file_path)
    rna_seqs = df["RNAp_seq"]

    print("Running make rna features parallel")
    st = time()
    result_df = make_rna_features_parallel(rna_seqs)
    et = time()
    print(f"Took: {et - st} seconds")  # Took: 58.083433628082275 seconds
    result_df.to_csv("rna_p_new_features.csv", index=False)


def checks():
    # res = extract_base_pairs("((((((((....(((((((((...)))))))))))))))))((((..(((((((...)))))))..))))...(((.(((((.((.(((...))).)).))))).)))...")
    # tr = run_RNAfold_as_webtool("AGGTGTGTGAACCCGCGCGCGCGCG")
    # tr = run_RNAfold("AGGTGTGTGAACCCGCGCGCGCGCG")
    # run_RNAfold("AGGTGTGTGAACCCGCGCGCGCGCG")
    # res = get_prob_for_seq("AGGTGTGTGAACCCGCGCGCGCGCG")
    # prob_dict = base_pair_probabilities("dot.ps")
    # bp_ener = sum_base_pair_energy('VI')
    # res = run_RNAeval("CGUUUGUUUUUUUGGUGGCGAUGGUCGCCACCAAACAAACGGCCUAGUUCUCGAUGGUUGAGAAAAAGGCUUCCAUUGACCGAAGUCGUCUCGCGUCUAUGGUUUAUGACAAGAAGAU", "(((((((.....(((((((((...))))))))).)))))))((((..(((((((...)))))))..))))...(((.(((((.((.(((...))).)).))))).)))..........")
    # r = sum_base_pair_energy(res, "III")

    # import os  # to get the direction of the csv file
    # # Get the current directory of your script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # # Construct the relative path to the CSV file
    csv_file_path = os.path.join(script_dir, '..', '..', 'data/rna_p_data.csv')

    #
    df = pd.read_csv(csv_file_path)
    # # df = pd.read_csv(r"..data")
    rna_seqs = df["RNAp_seq"]
    rna_seqs = rna_seqs.tail(5)

    # df = get_rna_form(rna_seqs, list(range(82, 203, 10)) + [554])
    # df = rna_features_by_windows(rna_seqs)
    # df.to_csv("windows.csv")

    print("Running make rna features not paralel")
    st_u = time()
    result_df_not_par = make_rna_features(rna_seqs)
    et_u = time()
    print(f"Took: {et_u - st_u} seconds")   # Took: 277.6278851032257 seconds
    result_df_not_par.to_csv("not_par.csv")

    print("Running mae rna features paralel")
    st = time()
    result_df = make_rna_features_parallel(rna_seqs)
    et = time()
    print(f"Took: {et - st} seconds")   # Took: 58.083433628082275 seconds
    result_df.to_csv("par.csv")

    print("done")
    # make_rna_features(rna_seqs)
    # get_match_ratio("CGUUUGUUUUUUUGGUGGCGAUGGUCGCCACCAAACAAACGGCCUAGUUCUCGAUGGUUGAGAAAAAGGCUUCCAUUGACCGAAGUCGUCUCGCGUCUAUGGUUUAUGACAAGAAGAU", CONSENSUS_POSITIONS_3_STEM_LOOPS)
    # get_prob_df(seq, seq_end_idx=[10, 20])
    # mfe_centroid_comparison = get_mfe_centroid_comparison_df(rna_seqs, [120, 130, 140, 150, 200, 300, 350, 554])
    # return mfe_centroid_comparison

def check_output():
    df = pd.read_csv("rna_p_new_features.csv")
    print(df.shape)


if __name__ == '__main__':
    generate_features_csv()
    check_output()
    # checks()

    # Now, you can work with the 'df' DataFrame as needed
    # For example, you can access columns and perform data analysis:
    # print(df.head())  # Print the first few rows of the DataFrame

# features to extract from RNAfold files and RNAeval output

# 1) take rna_p[3:32] and make dictionary of the indices:                   V
# Interior loop (  4, 33) GC; (  5, 32) GC:  -330
# Interior loop (  5, 32) GC; (  6, 31) UA:  -220
# Interior loop (  6, 31) UA; (  7, 30) AU:  -130
# Interior loop (  7, 30) AU; (  9, 28) CG:   120
# Interior loop (  9, 28) CG; ( 10, 27) UA:  -210
# Interior loop ( 10, 27) UA; ( 11, 26) GC:  -210
# Interior loop ( 11, 26) GC; ( 12, 24) GC:    50
# Interior loop ( 12, 24) GC; ( 13, 23) CG:  -340
# Interior loop ( 13, 23) CG; ( 14, 22) UA:  -210
# Interior loop ( 14, 22) UA; ( 15, 21) UG:  -130
# Hairpin  loop ( 15, 21) UG              :   550


# run RNAfold on entire seq and take using RNAeval energy sum of extended alpha beta loop (SL 4)
# stem-loop IV seq[3:125]
# Interior loop (  4,126) GC; (  5,125) GC:  -330
# Interior loop (  5,125) GC; (  6,124) UA:  -220
# Interior loop (  6,124) UA; (  7,123) AU:  -130
# Interior loop (  7,123) AU; (  8,122) AU:   -90
# Interior loop (  8,122) AU; (  9,121) CG:  -220
# Interior loop (  9,121) CG; ( 11,119) GC:    40
# Interior loop ( 11,119) GC; ( 12,118) GC:  -330
# Interior loop ( 12,118) GC; ( 14,116) UA:   120
# Interior loop ( 14,116) UA; ( 15,115) UA:   -90
# Interior loop ( 15,115) UA; ( 17,114) AU:   250
# Interior loop ( 17,114) AU; ( 18,113) GC:  -210
# Interior loop ( 18,113) GC; ( 19,112) CG:  -340
# Interior loop ( 19,112) CG; ( 20,111) AU:  -210
# Interior loop ( 20,111) AU; ( 21,110) GC:  -210
# Interior loop ( 21,110) GC; ( 22,109) AU:  -240
# Interior loop ( 22,109) AU; ( 23,108) GC:  -210
# Interior loop ( 23,108) GC; ( 24,107) CG:  -340
# Interior loop ( 24,107) CG; ( 25,106) GC:  -240
# Interior loop ( 25,106) GC; ( 27,105) AU:   140
# Interior loop ( 27,105) AU; ( 28,104) GC:  -210
# Interior loop ( 41, 82) GC; ( 42, 81) UA:  -220
# Interior loop ( 42, 81) UA; ( 43, 80) UA:   -90
# Interior loop ALPHA_RANGE GC:  -210
# Hairpin  loop ( 89, 96) GC              :   470
# Multi    loop ( 28,104) GC              :   360

# run RNAfold on entire seq and look at the C rich area, compare to consensus, take probabilities and energies.
# G rich area seq[216:222]  - should be in the loop and not base-paired

