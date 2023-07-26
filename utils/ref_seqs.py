from Bio import Entrez
import Bio
from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
from Bio.Align import substitution_matrices
import os
import pandas as pd
from typing import List


def download_reference_fasta(reference, output_path):
    Entrez.email = 'your_email@example.com'  # Replace with your email address

    try:
        handle = Entrez.efetch(db='nucleotide', id=reference, rettype='fasta', retmode='text')
        with open(output_path, 'w') as f:
            f.write(handle.read())
        print("FASTA file downloaded successfully.")
    except Exception as e:
        print("Failed to download the FASTA file:", str(e))


def get_fastas():
    if not os.path.exists("fastas"):
        os.mkdir("fastas")

    df = pd.read_excel("journal.pgen.1009919.s006.xlsx")
    ecoli_ref_lst = df[df["Genus"] == "Escherichia"]["Reference"]
    for r in ecoli_ref_lst:
        download_reference_fasta(r, os.path.join("fastas", r + ".fasta"))

def scoring_function(c1, c2):
    if c1 == c2:
        return 1  # Match score
    elif c1 == "-" or c2 == "-":
        return -10  # Gap score
    else:
        return -1  # Mismatch score


def find_promoter_alignment_2(fasta_file, prom_seq):
    gap_open_penalty = -3
    gap_extend_penalty = -1

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        # sequence = "AAAAGGGGTTGTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTAAAAGGGGTTG"

        substitution_matrix = [
            [1, -1, -1, -1],  # A
            [-1, 1, -1, -1],  # C
            [-1, -1, 1, -1],  # G
            [-1, -1, -1, 1],  # T
            [0, -10, -10, 0]  # Gap
        ]

        # substitution_matrix = [
        #     [1 if i == j else -1 for j in "ACGT"]
        #     for i in "ACGT"
        # ]

        # Perform pairwise alignment using Needleman-Wunsch algorithm
        # alignments = pairwise2.align.globalxd(prom_seq, sequence, penalize_extend_when_opening=True, penalize_end_gaps=True)
        # alignments = pairwise2.align.localds(prom_seq, sequence, substitution_matrix, -5, -5)

        # sub_mat = substitution_matrices.load("MEGABLAST")
        sub_mat = substitution_matrices.load("BLOSUM62")
        alignments = pairwise2.align.localds(prom_seq, sequence, sub_mat,
                                             gap_open_penalty, gap_extend_penalty)
        # alignments = pairwise2.align.globalms(prom_seq, sequence, 1, -1, -10, -10, score_only=False,
        #                                       one_alignment_only=True, aligner='local', penalize_end_gaps=False,
        #                                       gap_char=["-"], force_generic=True)

        # Find the alignment with the highest score
        best_alignment = max(alignments, key=lambda x: x.score)

        # Bio.pairwise2.print_matrix()

        print("best alignment:")
        print(format_alignment(*best_alignment, full_sequences=False))

        # Extract alignment information
        alignment_score = best_alignment.score
        target_alignment = best_alignment.seqA
        sequence_alignment = best_alignment.seqB

        # Print alignment information
        print("Alignment found in sequence:", record.id)
        print("Alignment score:", alignment_score)
        print("Alignment target sequence:", target_alignment)
        print("Alignment sequence:", sequence_alignment)
        print()


def get_promoter_alignment(fasta_file, prom_seq):
    gap_open_penalty = -3
    gap_extend_penalty = -1

    # assuming there is only one seq in the fasta
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)

        sub_mat = substitution_matrices.load("NUC.4.4")
        pairwise2.MAX_ALIGNMENTS = 10
        alignments = pairwise2.align.localds(prom_seq, sequence, sub_mat,
                                             gap_open_penalty, gap_extend_penalty)

        # Find the alignment with the highest score
        print(f"Found {len(alignments)} alignments for {fasta_file} and prom: {prom_seq}")
        best_alignment = max(alignments, key=lambda x: x.score)
        return best_alignment


def get_align_fastas_dict(fastas_dir, prom_seq):
    align_dict = {}
    for fasta in os.listdir(fastas_dir):
        alignment = get_promoter_alignment(os.path.join(fastas_dir, fasta), prom_seq)
        align_dict[fasta] = alignment

    return align_dict


RNA_P = "gcaaacaaaaaaaccaccgctaccagcggtggtttgtttgccggatcaagagctaccaactctttttccgaaggtaactggcttcagcagagcgcagataccaaatactgttcttctagtgtagccgtagttaggccaccacttcaagaactctgtagcaccgcctacatacctcgctctgctaatcctgttaccagtggctgctgccagtggcgataagtcgtgtcttaccgggttggactcaagacgatagttaccggataaggcgcagcggtcgggctgaacggggggttcgtgcacacagcccagcttggagcgaacgacctacaccgaactgagatacctacagcgtgagctatgagaaagcgccacgcttcccgaagggagaaaggcggacaggtatccggtaagcggcagggtcggaacaggagagcgcacgagggagcttccagggggaaacgcctggtatctttatagtcctgtcgggtttcgccacctctgacttgagcgtcgatttttgtgatgctcgtcaggggggcggagcctatggaaa".upper()
PROMOTER_RNA_P = "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT"


def get_alignment(fasta_file, seq):
    ali = get_promoter_alignment(fasta_file, seq)
    return ali


def find_appropriate_combo(prom_alis: List[Bio.pairwise2.Alignment], seq_alis: List[Bio.pairwise2.Alignment]):
    dist = 4000
    best_combo = [0, 0]
    for pa in prom_alis:
        for sa in seq_alis:
            curr_dist = abs(pa.start - sa.start)
            if curr_dist < dist:
                dist = curr_dist
                best_combo[0] = pa
                best_combo[1] = sa

    return best_combo


def get_ali_prom_and_seq(fasta_file, prom, seq):
    gap_open_penalty = -3
    gap_extend_penalty = -1

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)

        sub_mat = substitution_matrices.load("NUC.4.4")
        pairwise2.MAX_ALIGNMENTS = 10
        prom_alignments = pairwise2.align.localds(prom, sequence, sub_mat,
                                             gap_open_penalty, gap_extend_penalty)
        seq_alignments = pairwise2.align.localds(seq, sequence, sub_mat,
                                                  gap_open_penalty, gap_extend_penalty)

        # Find the alignment with the highest score
        # print(f"Found {len(alignments)} alignments for {fasta_file} and prom: {prom_seq}")
        # best_alignment = max(alignments, key=lambda x: x.score)
        best_combo = find_appropriate_combo(prom_alignments, seq_alignments)
        return best_combo[0], best_combo[1]     # promotor_align, seq_align


def match_promoter_vs_rna_p():
    folder = "fastas"
    for fasta in os.listdir(folder):
        fasta_path = os.path.join(folder, fasta)
        # prom_ali = get_alignment(fasta_path, PROMOTER_RNA_P)
        # rna_p_ali = get_alignment(fasta_path, RNA_P)
        prom_ali, seq_ali = get_ali_prom_and_seq(fasta_path, PROMOTER_RNA_P, RNA_P)
        try:
            print(f"{fasta}: alignment locations: promotor: {prom_ali.start}, rna_p seq: {seq_ali.start}, dist: {seq_ali.start - prom_ali.start}, gap prom end to seq start: {seq_ali.start - prom_ali.end}")
        except Exception as ex:
            print(str(ex))


def parse_fastas():
    promoter_sequence_p = "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT"
    res_d = get_align_fastas_dict("fastas", promoter_sequence_p)
    df = pd.DataFrame({"fasta": res_d.keys(),
                       "alignment_score": [al.score for al in res_d.values()],
                       "start_i": [al.start for al in res_d.values()],
                       "end_i": [al.end for al in res_d.values()],
                       "al_print": [format_alignment(*al, full_sequences=False) for al in res_d.values()],
                       })
    df.to_csv("fastas_pairwise_RNAp.csv")


def check_low_dist_fastas():
    fastas_dir = "fastas"

    with open(r"analysis\dist_check.txt", "r") as f:
        data = [l.split(", ") for l in f.readlines()]

    interesting_fastas = []
    for l in data:
        for sp in l:
            if "dist" in sp:
                curr_dist = int(sp.split(": ")[1])
                if curr_dist < 100:
                    interesting_fastas.append(l[0].split(": ")[0])

    print(f"Found {len(interesting_fastas)}")
    print("Now checking alignment")

    align_dict = {}
    for f in interesting_fastas:
        alignment = get_promoter_alignment(os.path.join(fastas_dir, f), PROMOTER_RNA_P)
        align_dict[f] = alignment

    for f in align_dict:
        print(f)
        print(format_alignment(*align_dict[f]))


# Todo:
#  1) run MSA after pairwise
#  2) try to consider specific length of promoter and make pairwise
#  3) try to cluster promoters to groups


if __name__ == '__main__':
    # Usage example:
    # fasta_file = r"fastas\AF158026.1.fasta"
    # promoter_sequence_p = "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT"
    #
    # allegedly_promotor = "TGTTCCTGTTTCACTTTCAGTCTCAAAGCGAACCTGGATGCTGTTCTGGAGTTCTTCCGCGAG"
    #
    # fastas_dir = "fastas"
    # for fasta_file in os.listdir(fastas_dir)[:10]:
    #     find_promoter_alignment_2(os.path.join(fastas_dir, fasta_file), promoter_sequence_p)

    # match_promoter_vs_rna_p()
    # check_low_dist_fastas()
    parse_fastas()

