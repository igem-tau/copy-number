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

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)

        sub_mat = substitution_matrices.load("BLOSUM62")
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


RNA_P = "gcaaacaaaaaaaccaccgctaccagcggtggtttgtttgccggatcaagagctaccaactctttttccgaaggtaactggcttcagcagagcgcagataccaaatactgttcttctagtgtagccgtagttaggccaccacttcaagaactctgtagcaccgcctacatacctcgctctgctaatcctgttaccagtggctgctgccagtggcgataagtcgtgtcttaccgggttggactcaagacgatagttaccggataaggcgcagcggtcgggctgaacggggggttcgtgcacacagcccagcttggagcgaacgacctacaccgaactgagatacctacagcgtgagctatgagaaagcgccacgcttcccgaagggagaaaggcggacaggtatccggtaagcggcagggtcggaacaggagagcgcacgagggagcttccagggggaaacgcctggtatctttatagtcctgtcgggtttcgccacctctgacttgagcgtcgatttttgtgatgctcgtcaggggggcggagcctatggaaa"
PROMOTER_RNA_P = "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT"


def get_alignment(fasta_file, seq):
    ali = get_promoter_alignment(fasta_file, seq)
    print("done")


def match_promoter_vs_rna_p():
    folder = "fastas"
    for fasta in os.listdir(folder):
        rna_p_ali = get_alignment(os.path.join(folder, fasta), RNA_P)
        prom_ali = get_alignment(os.path.join(folder, fasta), PROMOTER_RNA_P)
        print(f"matching promoter and rna p in {fasta}")


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

    # main()


