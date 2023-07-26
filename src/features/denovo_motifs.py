import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import get_current_file_parent_path


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')

HOMER_HIGH_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'high.motifs')

HOMER_LOW_MOTIF_PATH = Path(DATA_PATH, 'homer_motifs', 'low.motifs')


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
                score += np.log2(float(pssm.loc[i, nt])/0.25)
            scores.append(score)
        max_score = max(scores)
    return max_score


def score_denovo_motifs(df):
    denovo_motifs_features = {}
    high_motifs_dict = get_denovo_motifs_pssms(HOMER_HIGH_MOTIF_PATH)
    low_motifs_dict = get_denovo_motifs_pssms(HOMER_LOW_MOTIF_PATH)
    def seq_score_denovo_motifs(seq):
        for motif_name in high_motifs_dict.keys():
            score = calc_max_pssm_score_sliding_window(seq, high_motifs_dict[motif_name])
            denovo_motifs_features[motif_name + '_denovo_HIGH'] = score
        for motif_name in low_motifs_dict.keys():
            score = calc_max_pssm_score_sliding_window(seq, low_motifs_dict[motif_name])
            denovo_motifs_features[motif_name + '_denovo_LOW'] = score
        return pd.Series(denovo_motifs_features)
    return pd.DataFrame(df.apply(seq_score_denovo_motifs))


if __name__ == '__main__':
    example_seq = 'A' * 30
    print(score_denovo_motifs(example_seq))
