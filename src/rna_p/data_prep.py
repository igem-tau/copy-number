from src.data_prep.pre_process import get_RNAi_data


DNA_PROMOTER_i = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
DNA_p = 'GCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTTCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCGTGAGCTATGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGCAGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTCCTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAGCCTATGGAAA'


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


def play_with_p_RNA():
    overlap = get_complement(DNA_PROMOTER_i, mode='dna')[-1::-1]
    assert overlap in DNA_p, "Promoter i complement not in DNA p"
    start_idx = DNA_p.find(overlap)
    end_idx = start_idx + len(overlap)

    df = get_RNAi_data()
    df["complement_reverse"] = df['Promoter Sequence (-35 to +1)'].apply(lambda x: get_complement(x, mode='dna')[-1::-1])
    df["RNAp_seq"] = (DNA_p[:start_idx] + df["complement_reverse"] + DNA_p[end_idx:]).apply(lambda x: get_complement(x, mode='rna'))
    df.drop(columns=["complement_reverse"], inplace=True)
    df.to_csv(r"..\..\data\rna_p_data.csv")


if __name__ == '__main__':
    play_with_p_RNA()