import os
import pandas as pd
from src.data_prep.pre_process import get_RNAi_data


DNA_PROMOTER_i = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
DNA_p = 'GCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTTCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCGTGAGCTATGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGCAGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTCCTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAGCCTATGGAAA'

# This is in forward strand (like DNA_p)
DNA_i = 'TGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTT'

# This is the generated RNA i
RNA_i = 'UUGUCAUAAACCAUAGACGCGAGACGACUUCGGUCAAUGGAAGCCUUUUUCUCAACCAUCGAGAACUAGGCCGUUUGUUUGGUGGCGACCAUCGCCACCAAAAAAACAAACGUUCGUCGUCUAAUGCGCGUCUUUUUUUCCUAGAGU'

RNA_i_shortened = 'UUGUCAUAAACCAUAGACGCGAGACGACUUCGGUCAAUGGAAGCCUUUUUCUCAACCAUCGAGAACUAGGCCGUUUGUUUGGUGGCGACCAUCGCCACCAAAA'


RNA_i = "UUGUCAUAAACCAUAGACGCGAGACGACUUCGGUCAAUGGAAGCCUUUUUCUCAACCAUCGAGAACUAGGCCGUUUGUUUGGUGGCGACCAUCGCCACCAAAAAAACAAACGUUCGUCGUCUAAUGCGCGUCUUUUUUUCCUAGAGU"


def get_complement(seq, mode):
    res = ""
    for n in seq:
        if n == "G":
            res += "C"
        elif n == "C":
            res += "G"
        elif n == "A":
            if mode == 'rna':
                res += "U"
            elif mode == 'dna':
                res += "T"
        elif n == "T":
            res += "A"
    return res


def get_wild_type():
    return get_complement(DNA_p, mode='rna')


def generate_p_RNA():
    overlap = get_complement(DNA_PROMOTER_i, mode='dna')[-1::-1]
    assert overlap in DNA_p, "Promoter i complement not in DNA p"
    start_idx = DNA_p.find(overlap)
    end_idx = start_idx + len(overlap)

    df = get_RNAi_data()
    df["complement_reverse"] = df['Promoter Sequence (-35 to +1)'].apply(lambda x: get_complement(x, mode='dna')[-1::-1])
    df["RNAp_seq"] = (DNA_p[:start_idx] + df["complement_reverse"] + DNA_p[end_idx:]).apply(lambda x: get_complement(x, mode='rna'))
    df.drop(columns=["complement_reverse"], inplace=True)
    df.to_csv(r"..\..\data\rna_p_data.csv")


def load_data():
    if os.path.exists(r"..\..\data\rna_p_data.csv"):
        df = pd.read_csv(r"..\..\data\rna_p_data.csv")
        return df
    raise Exception(f"You need to make the csv with RNAp data first, call generate_p_RNA")


if __name__ == '__main__':
    generate_p_RNA()
    # pass