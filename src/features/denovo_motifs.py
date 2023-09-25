import numpy as np
import pandas as pd
from pathlib import Path
from src.consts import RNA_TYPE_CONST
from src.utils import get_current_file_parent_path, is_feature_selected
from tqdm import tqdm
from typing import List, Optional

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')

# TODO - pass rna_rtype through function's parameters
if RNA_TYPE_CONST['RNA'] == 'p':
    HOMER_HIGH_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'high_pRNA.motifs')
    HOMER_LOW_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'low_pRNA.motifs')

elif RNA_TYPE_CONST['RNA'] == 'i':
    HOMER_HIGH_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'high_iRNA.motifs')
    HOMER_LOW_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'low_iRNA.motifs')


def get_denovo_motifs_pssms(homer_output_file_loc):
    with open(homer_output_file_loc, 'r') as f:
        motifs = f.read()
    motifs = motifs.split('>')

    motifs_dict = {}
    for motif in motifs:
        if motif and not ('NNNNNNNN' in motif):
            table_motif = motif.strip().split('\n', 1)[1]
            motif_name = motif.split('\t')[0]
            motif_df = pd.DataFrame(
                [x.split('\t') for x in table_motif.split('\n')],
                columns=['A', 'C', 'G', 'T'],
            )
            motif_df = motif_df.fillna(0)
            motif_df = motif_df.replace('', 0)
            motifs_dict[motif_name] = motif_df
    return motifs_dict


def calc_max_pssm_score_sliding_window(seq: str, pssm: pd.DataFrame) -> float:
    max_score = 0
    scores = []
    num_windows = len(seq) - len(pssm) + 1
    if num_windows > 0:
        for j in range(num_windows):
            score = 0
            for i in range(len(pssm)):
                nt = seq[i + j]
                score += np.log2(float(pssm.loc[i, nt]) / 0.25)
            scores.append(score)
        max_score = max(scores)
    return max_score


def score_denovo_motifs(sequences: 'pd.Series[str]', selected_features: Optional[List[str]]):
    denovo_motifs_features = {}
    high_motifs_dict = get_denovo_motifs_pssms(HOMER_HIGH_MOTIF_PATH)
    low_motifs_dict = get_denovo_motifs_pssms(HOMER_LOW_MOTIF_PATH)

    def seq_score_denovo_motifs(seq):
        for motif_name in high_motifs_dict.keys():
            feature_name = f'{motif_name}_denovo_HIGH'
            if is_feature_selected(feature_name, selected_features):
                score = calc_max_pssm_score_sliding_window(seq, high_motifs_dict[motif_name])
                denovo_motifs_features[feature_name] = score
        for motif_name in low_motifs_dict.keys():
            feature_name = f'{motif_name}_denovo_LOW'
            if is_feature_selected(feature_name, selected_features):
                score = calc_max_pssm_score_sliding_window(seq, low_motifs_dict[motif_name])
                denovo_motifs_features[feature_name] = score
        return pd.Series(denovo_motifs_features)

    tqdm.pandas(desc='score denovo motifs')
    return pd.DataFrame(sequences.progress_apply(seq_score_denovo_motifs))


if __name__ == '__main__':
    seq1 = 'A' * 30
    seq2 = 'C' * 30
    seq3 = 'G' * 30
    seq4 = 'T' * 30
    seq5 = 'TCAATCCTTT' + 'T' * 20
    seq6 = 'TCAATCCTTT' + 'TAACTGCT' + 'CAGCAGCATACG'
    print(score_denovo_motifs(pd.Series([seq1, seq2, seq3, seq4, seq5, seq6], name='sequence'), None))
