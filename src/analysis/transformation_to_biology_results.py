import numpy as np
import pandas as pd
from src.utils import get_current_file_parent_path
from pathlib import Path
import plotly
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

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


def fit_func_option_1(raw_pcn, a, b):
    return a * np.exp(raw_pcn) + b


def fit_func_option_2(raw_pcn, a, b, c):
    return a * raw_pcn ** b + c


def fit_func_option_3(raw_pcn, a, b, c):
    return a * raw_pcn ** (b / raw_pcn) + c


def fit_func_option_4(raw_pcn, a, b):
    return a * np.log2(raw_pcn) + b


def scatter_plot_raw_target(raw_traget_pcn_df):
    fig = px.scatter(raw_traget_pcn_df, x='x', y='y', color='type',
                     labels={'x': 'raw copy number', 'y': 'ddPCR / qPCR'})
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

    raw_target_pcn_df = df = pd.concat((
        pd.DataFrame(
            {'x': concat_ddpcr[RAW_PCN_COLUMN_NAME], 'y': concat_ddpcr[TARGET_PCN_COLUMN_NAME], 'type': 'ddPCR'}),
        pd.DataFrame(
            {'x': concat_qpcr[RAW_PCN_COLUMN_NAME], 'y': concat_qpcr[TARGET_PCN_COLUMN_NAME], 'type': 'qPCR'})), axis=0)

    fig = scatter_plot_raw_target(raw_target_pcn_df)

    option_1_popt, _ = curve_fit(fit_func_option_1, raw_target_pcn_df['x'], raw_target_pcn_df['y'])
    option_2_popt, _ = curve_fit(fit_func_option_2, raw_target_pcn_df['x'], raw_target_pcn_df['y'])
    option_3_popt, _ = curve_fit(fit_func_option_3, raw_target_pcn_df['x'], raw_target_pcn_df['y'])
    option_4_popt, _ = curve_fit(fit_func_option_4, raw_target_pcn_df['x'], raw_target_pcn_df['y'])

    option_5_z = np.polyfit(raw_target_pcn_df['x'], raw_target_pcn_df['y'], 3)
    fit_func_option_5 = np.poly1d(option_5_z)


    x_space = pd.Series(np.linspace(raw_target_pcn_df['x'].min(), raw_target_pcn_df['x'].max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: article_fit_func(x))), name='123*x^0.37-28'))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: fit_func_option_1(x, *option_1_popt))), name='a*exp(x)+b'))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: fit_func_option_2(x, *option_2_popt))), name='a*x^b+c'))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: fit_func_option_3(x, *option_3_popt))),
                             name='a*x^(b/raw_pcn)+c'))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: fit_func_option_4(x, *option_4_popt))), name='a*log2(x)+b'))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: fit_func_option_5(x))), name='3rd deg polynom'))

    fig.show()
    plotly.offline.plot(fig, filename=str(Path(DATA_PATH.parent, 'raw_pcn_fit_search.html')))

    # print(spearmanr(bio_p_cn, model_p_cn_trans))
    # fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='', showlegend=False))
