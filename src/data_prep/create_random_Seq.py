from Bio import SeqIO
import random
import pandas as pd
def sample_sequences(fasta_file, num_samples, window_len):
    df = pd.DataFrame()
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            sequence_length = len(sequence)
            sampled_sequences = []
            for _ in range(num_samples):
                random_position = random.randint(window_len, sequence_length - window_len)
                start_index = max(0, random_position - (window_len // 2))
                end_index = min(sequence_length, random_position + (window_len // 2) + 1)
                sampled_sequence = sequence[start_index:end_index]
                sampled_sequences.append(sampled_sequence)
            df[record.name]=sampled_sequences
    df.to_csv('random_seq.scv',index_label=False)




sampled_sequences = sample_sequences(r"GCF_000005845.2_ASM584v2_genomic.fna", num_samples=1000, window_len=40)



