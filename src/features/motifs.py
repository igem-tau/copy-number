import numpy as np
import pandas as pd
from pathlib import Path
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO
from src.utils import get_current_file_parent_path, is_feature_selected
from tqdm import tqdm
from typing import List, Optional

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
INTERACT_MEME = Path(DATA_PATH, 'motifs', 'dpinteract.meme')
SWISS_MEME = Path(DATA_PATH, 'motifs', 'SwissRegulon_e_coli.meme')


# generate motifs dictionary
def generate_motif_dict(selected_features: 'Optional[List[str]]'):
    meme_files = [INTERACT_MEME, SWISS_MEME]
    motifs = {}
    motifs_num = 0

    for meme_file in meme_files:
        with MotifFile(meme_file) as motif_file:
            motif = motif_file.read()
            while motif:
                motif_name = motif.accession.decode()
                if (is_feature_selected(f'{motif_name}_pv', selected_features) or is_feature_selected(
                        f'{motif_name}_score', selected_features)):
                    motifs_num += 1
                    motifs[motif_name] = motif
                motif = motif_file.read()

    assert motifs_num == len(motifs)
    return motifs, motif_file  # TODO -why is only (and always) the later motif_file returned


def calc_motifs_pv(seqs: 'pd.Series[str]', selected_features: 'Optional[List[str]]') -> pd.DataFrame:
    num_sequences = len(seqs)
    motifs, motif_file = generate_motif_dict(selected_features)
    fimo = FIMO(both_strands=True)  # , threshold=1e-3)

    motifs_dict = {}
    for motif_name, selected_motif in tqdm(motifs.items(), desc='motifs'):
        pv_feature_name = f'{motif_name}_pv'
        score_feature_name = f'{motif_name}_score'

        if is_feature_selected(pv_feature_name, selected_features):
            motifs_dict[pv_feature_name] = np.ones(num_sequences)
        if is_feature_selected(score_feature_name, selected_features):
            motifs_dict[score_feature_name] = np.zeros(num_sequences)

        for i, seq in enumerate(seqs):
            pattern = fimo.score_motif(selected_motif, [Sequence(seq)], motif_file.background)
            sorted_matches = sorted(pattern.matched_elements, key=lambda m: (m.pvalue, -m.score))
            if len(sorted_matches)>0:
                if is_feature_selected(pv_feature_name, selected_features):
                        motifs_dict[pv_feature_name][i] = sorted_matches[0].pvalue

                if is_feature_selected(score_feature_name, selected_features):
                        motifs_dict[score_feature_name][i] = sorted_matches[0].score

    return pd.DataFrame(motifs_dict)


if __name__ == '__main__':
    motifs_, motif_file_ = generate_motif_dict(selected_features=None)
    print("done")
