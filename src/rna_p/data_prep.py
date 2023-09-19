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


def generate_new_seqs():
    df = pd.read_csv(r"C:\Users\User1\IGEM\code\copy-number\data\rna_p_data.csv")
    highest_seq = df["RNAp_seq"].loc[365]   # copy number: 683.436
    ws = highest_seq

    # attempts to make stem stronger - adding GCs
    ns1 = ws[:100] + "GCGCCGC" + ws[107:163] + "GCGGCG" + ws[169:]   # using RNAfold - destroy first circle
    ns2 = ns1[:113] + "AGCGC" + ns1[118:148] + "UGCGCGCC" + ns1[156:]   # using RNAfold - destroy second circle
    ns3 = ns2[:126] + "CGCCACCGCGGUGGCG" + ns2[142:]    # using RNAfold - make tip stronger

    # attempts to destroy stem
    ns4 = ws[:88] + "UACUUA" + ws[94:174] + "CCGAAC" + ws[180:] # destroyed pre-first-cycle strong connection
    ns4_a = ws[:83] + "CGCAU" + ws[88:180] + "CGAAU" + ws[185:] # destroyed pre-first-cycle strong connection
    ns5 = ws[:107] + "UAUCUAG" + ws[114:156] + "UCUAGGC" + ws[163:] # attempt to destroy post-first-cycle connection - not destroyed
    ns6 = ws[:107] + "CCCCCCC" + ws[114:156] + "UUUUUUU" + ws[163:] # attempt to destroy post-first-cycle connection - not destroyed

    # attempts to make stem stronger - adding AUs
    ns7 = ws[:100] + "AUAUAUA" + ws[107:163] + "AUAUAU" + ws[169:]
    ns8 = ns7[:113] + "AAUAU" + ns7[118:148] + "UAUAUAUA" + ns7[156:]  # using RNAfold - destroy second circle
    ns9 = ns8[:126] + "UAAUAUUAUAAUAUUA" + ns8[142:]  # using RNAfold - make tip stronger

    # mix adding CGs and AUs
    ns10 = ns7[:113] + "AGCGC" + ns7[118:148] + "UGCGCGCC" + ns7[156:]  # using RNAfold - destroy second circle
    ns11 = ns8[:126] + "CGCCACCGCGGUGGCG" + ns8[142:]  # using RNAfold - make tip stronger

    ndf = pd.DataFrame({"new_rna_p": [ns1, ns2, ns3, ns4, ns4_a, ns5, ns6, ns7, ns8, ns9, ns10, ns11]})
    ndf.to_csv("new_rna_p_seqs.csv", index=False)


if __name__ == '__main__':
    # generate_p_RNA()
    generate_new_seqs()
    # pass