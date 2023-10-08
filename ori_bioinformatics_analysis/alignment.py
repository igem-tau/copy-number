import Bio.Entrez, Bio.Align, Bio.SeqIO, Bio.SeqRecord, Bio.pairwise2, Bio.Seq

alignment_params={
        "sub_mat" : Bio.Align.substitution_matrices.load("NUC.4.4"),
        "gap_open_penalty" : -3,
        "gap_extend_penalty" : -1
    }

def fasta_file_to_seq(fasta_file):
    record = list(Bio.SeqIO.parse(fasta_file, "fasta"))[0]
    sequence = str(record.seq)
    return sequence

def align(seq, fasta_seq):
    alignments = Bio.pairwise2.align.localds(seq, fasta_seq, alignment_params["sub_mat"], alignment_params["gap_open_penalty"], alignment_params["gap_extend_penalty"])
    best_alignment = max(alignments, key=lambda x: x.score)
    return best_alignment

def print_alignment(best_alignment):
    print(Bio.pairwise2.format_alignment(*best_alignment, full_sequences=False))
