import numpy as np
import pandas as pd
from src.utils import get_current_file_parent_path
from pathlib import Path
import plotly
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr, linregress

RNA_TYPE = 'p'
RAW_PCN_COLUMN_NAME = 'Raw Copy Number'
TARGET_PCN_COLUMN_NAME = 'pcn'
DATA_PATH = Path(get_current_file_parent_path(__file__).parent.parent, 'data')
DATA_FILE_PATH = Path(DATA_PATH, f'RNA{RNA_TYPE}_with_Raw_PCN.csv')
ADDITIONAL_DATA_FILE_PATH = Path(DATA_PATH, 'biology_results', f'PCN RNA{RNA_TYPE} results.xlsx')


def is_promoter_sequence_valid(sequence):
    return '-' not in sequence and len(sequence) == 36


def join_data_with_article_sequences(article_data, additional_data, additional_promoter_col='promoter seq'):
    merged_data = pd.merge(article_data, additional_data, how='inner', left_on='Promoter Sequence (-35 to +1)',
                           right_on=additional_promoter_col)
    merged_data.rename(columns={additional_data.columns.values[-1]: TARGET_PCN_COLUMN_NAME}, inplace=True)
    return merged_data


def article_fit_func(raw_pcn):
    return 123 * raw_pcn ** 0.37 - 28


def scatter_plot_raw_target(raw_traget_pcn_df):
    fig = px.scatter(raw_traget_pcn_df, x='x', y='y', color='type',
                     labels={'x': 'log raw copy number', 'y': 'log ddPCR/qPCR'})
    fig.update_traces(marker_size=5)
    fig.show()
    return fig


if __name__ == '__main__':
    article_data = pd.read_csv(DATA_FILE_PATH, index_col=0)
    article_ddpcr_data = pd.read_excel(ADDITIONAL_DATA_FILE_PATH, sheet_name=f'ddPCR - RNA{RNA_TYPE}')[
        ['promoter seq', 'ddpcr']]
    our_bio_data = pd.read_excel(ADDITIONAL_DATA_FILE_PATH, sheet_name=f'Our qPCR - RNA{RNA_TYPE}')[
        [f'RNA{RNA_TYPE} promoter', 'PCN measured in Wet lab']]

    # filter invalid promoters (due to indel)
    our_bio_data = our_bio_data.loc[our_bio_data[f"RNA{RNA_TYPE} promoter"].apply(is_promoter_sequence_valid)]

    concat_ddpcr = join_data_with_article_sequences(article_data, article_ddpcr_data, 'promoter seq')
    concat_qpcr = join_data_with_article_sequences(article_data, our_bio_data, f'RNA{RNA_TYPE} promoter')

    raw_target_pcn_df = pd.concat((
        pd.DataFrame(
            {'x': concat_ddpcr[RAW_PCN_COLUMN_NAME], 'y': concat_ddpcr[TARGET_PCN_COLUMN_NAME], 'type': 'ddPCR'}),
        pd.DataFrame(
            {'x': concat_qpcr[RAW_PCN_COLUMN_NAME], 'y': concat_qpcr[TARGET_PCN_COLUMN_NAME], 'type': 'qPCR'})), axis=0)

    # raw_target_pcn_df = raw_target_pcn_df.query('x < 25')

    raw_target_pcn_df['x'] = raw_target_pcn_df['x'].apply(np.log)
    raw_target_pcn_df['y'] = raw_target_pcn_df['y'].apply(np.log)

    fig = scatter_plot_raw_target(raw_target_pcn_df)

    slope, intercept, pearson_corr, p_value, std_err = linregress(raw_target_pcn_df['x'], raw_target_pcn_df['y'])
    print(f'{slope=} {intercept=} {pearson_corr=} {p_value=} {std_err=}')
    spearman_corr = spearmanr(raw_target_pcn_df['y'], raw_target_pcn_df['x'])
    print('spearman', spearman_corr)

    x_space = pd.Series(np.linspace(raw_target_pcn_df['x'].min(), raw_target_pcn_df['x'].max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: slope * x + intercept)), name='linear regression'))

    # fig.add_trace(go.Scatter(x=x_space,
    #                          y=(x_space.apply(lambda x: np.exp(slope * x + intercept))),
    #                          name='pcn by linear regression'))

    fig.update_layout(title=dict(text=f'raw copy number power log fit'))
    fig.add_trace(go.Scatter(
        x=[raw_target_pcn_df['x'].min(), raw_target_pcn_df['x'].min()],
        y=[raw_target_pcn_df['y'].max(), raw_target_pcn_df['y'].max() - .3],
        mode="text",
        name='correlations metrics',
        text=[f'spearman: {spearman_corr.statistic:.3f}, p-value: {spearman_corr.pvalue:.3f}',
              f'pearson: {pearson_corr:.3f}, p-value: {p_value:.3f}'],
        textposition="top right"
    ))
    fig.show()
    plotly.offline.plot(fig, filename=str(Path(DATA_PATH, 'figures', 'raw_pcn_fit_search.html')))
