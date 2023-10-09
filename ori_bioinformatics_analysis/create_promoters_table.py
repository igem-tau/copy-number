import os
import pandas as pd
from consensus_sequences import RNAI_RNAP_dict, RNAI_RNAP_promoters_dict, promoter_seq_mapping, puc19_alignment_score_threshold
import alignment

def is_score_above_reverse_strand_null_model_threshold(seq_score, seq_name):
    return seq_score > puc19_alignment_score_threshold[seq_name]

def get_RNAI_RANP_alignmets_dicts(sequence):
    RNAI_RNAP_direction_score_dict={}
    RNAI_RNAP_alignment_dict={}
    RNAI_RNAP_alignment_direction_dict={}
    for seq_item in RNAI_RNAP_dict.items():
        seq_name, forward_backward_dict=seq_item
        forward_backward_score_dict={}
        for seq_item_direction in forward_backward_dict.keys():
            best_alignment = alignment.align(forward_backward_dict[seq_item_direction], sequence)
            best_score=best_alignment.score
            forward_backward_score_dict[seq_item_direction]=best_score
            RNAI_RNAP_alignment_direction_dict[seq_item_direction]={
                seq_name+"_score":best_score,
                seq_name+"_direction":seq_item_direction,
                seq_name+"_alignmentA":best_alignment.seqA[best_alignment.start:best_alignment.end],
                seq_name+"_alignmentB":best_alignment.seqB[best_alignment.start:best_alignment.end],
                seq_name+"_start":best_alignment.start,
                seq_name+"_end":best_alignment.end
            }
        best_score_item = max(forward_backward_score_dict.items(), key=lambda x: x[1])
        seq_direction, seq_score=best_score_item
        if is_score_above_reverse_strand_null_model_threshold(seq_score, seq_name):
            RNAI_RNAP_direction_score_dict[seq_name]=seq_direction
            RNAI_RNAP_alignment_dict[seq_name]=RNAI_RNAP_alignment_direction_dict[seq_direction]
    return RNAI_RNAP_direction_score_dict, RNAI_RNAP_alignment_dict
            
def extract_promoters(df, RNAI_RNAP_direction_score_dict, ref, sequence, promoter_length=100):
    for seq_item in RNAI_RNAP_promoters_dict.items():
            seq_name, forward_backward_dict=seq_item
            seq_direction=RNAI_RNAP_direction_score_dict[promoter_seq_mapping[seq_name]]
            RNA_type=promoter_seq_mapping[seq_name]
            if seq_direction=="forward":
                RNA_seq_start=df.loc[ref,RNA_type+"_start"]
                if RNA_seq_start-promoter_length >= 0:
                    df.loc[ref, seq_name]=sequence[RNA_seq_start-promoter_length:RNA_seq_start]
                else:
                    df.loc[ref, seq_name]=sequence[RNA_seq_start-promoter_length:]+sequence[:RNA_seq_start]
            else:
                RNA_seq_end=df.loc[ref, RNA_type+"_end"]
                if RNA_seq_end+promoter_length <= len(sequence):
                    df.loc[ref, seq_name]=sequence[RNA_seq_end:RNA_seq_end+promoter_length]
                else:
                    df.loc[ref, seq_name]=sequence[RNA_seq_end:]+sequence[:RNA_seq_end+promoter_length-len(sequence)]
            df.loc[ref, seq_name+"_direction"]=seq_direction

def extract_pylo_info(df, fasta_file, ref):
    info_arr=fasta_file.split("_")
    df.loc[ref, "species"]=info_arr[0]
    df.loc[ref, "genus"]=info_arr[1]
    df.loc[ref, "family"]=info_arr[2]
    df.loc[ref, "order"]=info_arr[3]
    
def extract_RNAI_RNAP(df, ref, RNAI_RNAP_alignment_dict):
    for seq_name in RNAI_RNAP_alignment_dict.keys():
        for alignment_item in RNAI_RNAP_alignment_dict[seq_name].items():
            name, value=alignment_item
            df.loc[ref, name]=value
            
def is_alignments_valid(RNAI_RNAP_direction_score_dict):
    # check if 2 alignments on different strands
    return len(RNAI_RNAP_direction_score_dict) == len(set(RNAI_RNAP_direction_score_dict.values())) and len(RNAI_RNAP_direction_score_dict)==2


def create_excel(plasmids_fastas_dir):
    df = pd.DataFrame(columns=["id","species","genus","family","order","RNAI_promoter","RNAI_promoter_direction","RNAP_promoter","RNAP_promoter_direction","RNAI_seq_score","RNAI_seq_start","RNAI_seq_end","RNAI_seq_alignmentA","RNAI_seq_alignmentB", "RNAI_seq_direction","RNAP_seq_score","RNAP_seq_start","RNAP_seq_end","RNAP_seq_alignmentA","RNAP_seq_alignmentB","RNAP_seq_direction"]).set_index("id")
    for fasta_file in os.listdir(plasmids_fastas_dir):#[:1]: #for testing
        ref=fasta_file
        print(fasta_file)
        sequence = alignment.fasta_file_to_seq(os.path.join(plasmids_fastas_dir, fasta_file))
        RNAI_RNAP_direction_score_dict, RNAI_RNAP_alignment_dict=get_RNAI_RANP_alignmets_dicts(sequence)

        if is_alignments_valid(RNAI_RNAP_direction_score_dict):
            extract_pylo_info(df, fasta_file, ref)
            extract_RNAI_RNAP(df, ref, RNAI_RNAP_alignment_dict)
            extract_promoters(df, RNAI_RNAP_direction_score_dict, ref, sequence)
                  
    df.to_csv("promoters_alignment.csv")
    df.to_excel("promoters_alignment_xl.xlsx")

if __name__ == "__main__":
    plasmids_fastas_dir = "fastas"
    create_excel(plasmids_fastas_dir)