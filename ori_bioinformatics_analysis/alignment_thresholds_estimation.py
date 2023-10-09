from consensus_sequences import RNAI_RNAP_seq_dict
import alignment

def score_pUC19(fasta_file):
    puc19_score_control = {}
    puc19_score_threshold={}
    puc19_score={}
    sequence = alignment.fasta_file_to_seq(fasta_file)
    for seq_item in RNAI_RNAP_seq_dict.items():
        seq_name, forward_backward_dict=seq_item
        for seq_item_direction in forward_backward_dict.keys():
            best_alignment = alignment.align(forward_backward_dict[seq_item_direction], sequence)
            print("Alignment of "+seq_name+" on "+seq_item_direction+" strand.")
            alignment.print_alignment(best_alignment)
            if seq_name not in puc19_score_control.keys():
                puc19_score_control[seq_name]={}    
            puc19_score_control[seq_name][seq_item_direction]=best_alignment.score
        puc19_score_threshold[seq_name]=min(puc19_score_control[seq_name].values())
        puc19_score[seq_name]=max(puc19_score_control[seq_name].values())
    return puc19_score_threshold, puc19_score

def get_null_model_alignment_threshold():
    #TODO: download ecoli genome fasta, sample 100bp x 1000 times, align to RNAI_RNAP_seq_dict values and return the stats
    pass


if __name__ == "__main__":
    fasta_file = b"data/pUC19.fa"
    other_strand_null_model_score, max_score=score_pUC19(fasta_file)
    print("Current null model (other strand) as a lower bound threshold for alignment:")
    print(other_strand_null_model_score)
    print("Max score of alignment (as an upper bound threshold for alignment):")
    print(max_score)
