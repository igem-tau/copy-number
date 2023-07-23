import logomaker
import pandas as pd
from pathlib import Path
from src.data_prep.pre_process import get_RNAp_data, get_RNAi_data
from src.features.pssm_feature import calc_pssm_matrix
from src.utils import get_current_file_parent_path

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
FIGURES_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data', 'figures')


def low_high_cp(df: pd.DataFrame):
    percentage = 0.2
    n = int(df.shape[0] * percentage)
    high_cp = df.nlargest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
    low_cp = df.nsmallest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
    return high_cp, low_cp


def plot_pssm(rna_type:str = 'p', save=False):
    RNA_data = get_RNAp_data() if rna_type == 'p' else get_RNAi_data()
    high_cp, low_cp = low_high_cp(RNA_data)

    high_logo = logomaker.Logo(calc_pssm_matrix(high_cp, False))
    high_logo.ax.set_title(f'PSSM for {rna_type}RNA - High copy number')
    high_logo.ax.set_ylabel('probabilities')
    high_logo.ax.set_xlabel('position')
    low_logo = logomaker.Logo(calc_pssm_matrix(low_cp, False))
    low_logo.ax.set_title(f'PSSM for {rna_type}RNA - Low copy number')
    low_logo.ax.set_ylabel('probabilities')
    low_logo.ax.set_xlabel('position')

    fig_high = calc_pssm_matrix(high_cp, False).plot.bar(stacked=True, title=f'{rna_type}RNA - High copy number', ylabel='probabilities', xlabel='position')
    fig_low = calc_pssm_matrix(low_cp, False).plot.bar(stacked=True, title=f'{rna_type}RNA - Low copy number', ylabel='probabilities', xlabel='position')

    if save:
        high_logo.fig.savefig(Path(FIGURES_PATH, f'{rna_type}RNA High copy number logo'))
        low_logo.fig.savefig(Path(FIGURES_PATH, f'{rna_type}RNA Low copy number logo'))
        fig_high.figure.savefig(Path(FIGURES_PATH, f'{rna_type}RNA - High copy number bar plot.jpg'))
        fig_low.figure.savefig(Path(FIGURES_PATH, f'{rna_type}RNA - Low copy number bar plot.jpg'))


if __name__=='__main__':
    plot_pssm('p', True)
    plot_pssm('i', True)


