from pydrive.auth import GoogleAuth
from google.colab import drive
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pymemesuite.common import MotifFile
from pymemesuite.fimo import FIMO
from pymemesuite.common import Sequenc
import numpy as np
import pandas as pd


# Todo: change it to use local files
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1vSxLDZb4FxIRY_ULW7gPe6ah2eUHkoTf'
download = drive.CreateFile({'id': file_id})
# Download the file to a local disc
download.GetContentFile('dpinteract.meme')
file_id = '18gsjran0xUjsQlQ6P6Ql7We_cZStmGZ6'
download = drive.CreateFile({'id': file_id})
# Download the file to a local disc
download.GetContentFile('SwissRegulon_e_coli.meme')


# generate motifs dictionary
def generate_motif_dict():
    motifs = {}
    motifs_num = 0
    with MotifFile("dpinteract.meme") as motif_file:
        motif = motif_file.read()
        while motif:
            motifs_num += 1
            motif_name = motif.accession.decode()
            motifs[motif_name] = motif
            motif = motif_file.read()

    with MotifFile("SwissRegulon_e_coli.meme") as motif_file:
        motif = motif_file.read()
        while motif:
            motifs_num += 1
            motif_name = motif.accession.decode()
            motifs[motif_name] = motif
            motif = motif_file.read()

    assert motifs_num == len(motifs)
    return motifs, motif_file


def calc_motifs_pv(seqs: 'pd.Series[str]') -> pd.DataFrame:
    motifs, _ = generate_motif_dict()
    fimo = FIMO(both_strands=True, threshold=1e-3)

    motifs_df = pd.DataFrame(data=np.ones((len(seqs), len(motifs.keys()))), columns=motifs.keys())
    for i, seq in enumerate(seqs):
      for selected_motif in motifs.values():
          pattern = fimo.score_motif(selected_motif, [Sequence(seq)], motif_file.background)
          for m in pattern.matched_elements:
              motifs_df.loc[i, selected_motif.accession.decode()] = m.pvalue
    return motifs_df