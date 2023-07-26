import numpy as np
import pandas as pd
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO


INTERACT_MEME = '../../data/motifs/dpinteract.meme'
SWISS_MEME = '../../data/motifs/SwissRegulon_e_coli.meme'


# generate motifs dictionary
def generate_motif_dict():
    motifs = {}
    motifs_num = 0
    with MotifFile(INTERACT_MEME) as motif_file:
        motif = motif_file.read()
        while motif:
            motifs_num += 1
            motif_name = motif.accession.decode()
            motifs[motif_name] = motif
            motif = motif_file.read()

    with MotifFile(SWISS_MEME) as motif_file:
        motif = motif_file.read()
        while motif:
            motifs_num += 1
            motif_name = motif.accession.decode()
            motifs[motif_name] = motif
            motif = motif_file.read()

    assert motifs_num == len(motifs)
    return motifs, motif_file


def calc_motifs_pv(seqs: 'pd.Series[str]') -> pd.DataFrame:
    motifs, motif_file = generate_motif_dict()
    fimo = FIMO(both_strands=True, threshold=1e-3)

    motifs_df = pd.DataFrame(data=np.ones((len(seqs), len(motifs.keys()))), columns=motifs.keys())
    for i, seq in enumerate(seqs):
      for selected_motif in motifs.values():
          pattern = fimo.score_motif(selected_motif, [Sequence(seq)], motif_file.background)
          for m in pattern.matched_elements:
              motifs_df.loc[i, selected_motif.accession.decode()] = m.pvalue
    return motifs_df