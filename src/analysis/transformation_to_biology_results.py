import numpy as np
import pandas as pd
from src.utils import get_current_file_parent_path
from pathlib import Path
import plotly.express as px
from scipy.stats import spearmanr, pearsonr

rna_type = 'p'
CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')

bio_data = pd.read_excel(Path(DATA_PATH, 'iGEM sequences table.xlsx'))
bio_data = bio_data[bio_data['Name of sequence '] != 'mut7']
RNA_p_data = pd.read_csv(Path(DATA_PATH.parent, 'copy_num_predictions_RNAp_XGBoost.csv'))

bio_data.fillna('', inplace=True)
RNA_p_mut = bio_data[bio_data['Name in order'].str.startswith(f'RNA{rna_type}')]

seqs = RNA_p_mut['RNAp promotor sequence']
bio_p_cn = RNA_p_mut['PCN in the lab'].values
model_p_cn = np.empty((len(bio_p_cn)))
for i, seq in enumerate(seqs):
    model_p_cn[i] = RNA_p_data[RNA_p_data['Promoter Sequence (-35 to +1)'] == seq]['Copy Number'].values



def func(x):
    return 123*x**(0.37) - 28

model_p_cn_trans = func(model_p_cn)

model_p_cn_trans = np.delete(model_p_cn_trans,[11, 4])
bio_p_cn = np.delete(bio_p_cn, [11, 4])

fig = px.scatter(x=bio_p_cn, y=model_p_cn_trans, labels={'x': 'Biology', 'y': 'Model'})
fig.show()

print(spearmanr(bio_p_cn, model_p_cn_trans))
# fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='', showlegend=False))
t = 3
