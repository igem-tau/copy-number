import matplotlib.pyplot as plt
from Bio import SeqIO
import random
import pandas as pd
import consensus_sequences as cc
import alignment
import warnings

def sample_sequences(fasta_file, num_samples, window_len,rna_type='tnp'):
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
    df.to_csv(f'data/random_seq_{rna_type}.scv',index_label=False)
    return df
def generate_lower_threshold(df_rand_chimera_score):
    return(df_rand_chimera_score['chimera score'].median())
def hist_plots(*hists):
    fig,ax=plt.subplots(len(hists),1,sharex=True)
    for i in range(len(hists)):
        ax[i].hist(hists[i][1])
        ax[i].set_title(f'{hists[i][0]} histogram')
    fig.tight_layout()
    plt.xlabel('Best alignment score from random ecoli sequence')
    fig.show()
def run(fasta_genome, num_samples):
    best_alignment_RNAI_FORWARD_l=[]
    best_alignment_RNAI_REVERSE_l=[]
    best_alignment_RNAP_FORWARD_l=[]
    best_alignment_RNAP_REVERSE_l=[]
    df_sequences={}
    best_alignment_rnap_l=[]
    best_alignment_rnai_l=[]
    RNAP_REVERSE=cc.RNAI_RNAP_dict['RNAP_seq']["reverse_complememt"]
    RNAP_FORWARD = cc.RNAI_RNAP_dict['RNAP_seq']["forward"]
    RNAI_REVERSE=cc.RNAI_RNAP_dict['RNAI_seq']["reverse_complememt"]
    RNAI_FORWARD = cc.RNAI_RNAP_dict['RNAI_seq']["forward"]
    win_lens=[['RNAI',len(RNAP_FORWARD)],['RNAP',len(RNAP_FORWARD)]]
    for i in win_lens:
        df_sequences[i[0]]=sample_sequences(fasta_genome, num_samples, int(i[1]))
    for seq in df_sequences['RNAI'].iloc[:,0]:
        best_alignment_RNAI_FORWARD = alignment.align(RNAI_FORWARD,seq).score
        best_alignment_RNAI_REVERSE = alignment.align(RNAI_REVERSE,seq).score
        best_alignment_rnai_l.append(max(best_alignment_RNAI_FORWARD,best_alignment_RNAI_REVERSE))
        best_alignment_RNAI_FORWARD_l.append(best_alignment_RNAI_FORWARD)
        best_alignment_RNAI_REVERSE_l.append(best_alignment_RNAI_REVERSE)


    for seq in df_sequences['RNAP'].iloc[:,0]:
        best_alignment_RNAP_FORWARD = alignment.align(RNAP_FORWARD,seq).score
        best_alignment_RNAP_REVERSE = alignment.align(RNAP_REVERSE, seq).score
        best_alignment_rnap_l.append(max(best_alignment_RNAP_REVERSE,best_alignment_RNAP_FORWARD))
        best_alignment_RNAP_FORWARD_l.append(best_alignment_RNAP_FORWARD)
        best_alignment_RNAP_REVERSE_l.append(best_alignment_RNAP_REVERSE)

    best_alignment_rnap=max(best_alignment_rnap_l)
    best_alignment_rnai=max(best_alignment_rnai_l)
    print(f'Best alignment from random ecoli sequence with RNAP is : {best_alignment_rnap}\nBest alignment from random ecoli sequence with RNAI is : {best_alignment_rnai}')
    hist_plots(['RNAI_FORWARD',best_alignment_RNAI_FORWARD_l],['RNAI_REVERSE',best_alignment_RNAI_REVERSE_l],['RNAP_FORWARD',best_alignment_RNAP_FORWARD_l],['RNAP_REVERSE',best_alignment_RNAP_REVERSE_l])




if __name__ =="__main__":
    warnings.simplefilter("ignore")
    fasta_genome=r"data/ecoli_genome_ref.fna"
    num_samples = 1000
    run(fasta_genome, num_samples)



