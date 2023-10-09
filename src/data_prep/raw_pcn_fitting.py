import numpy as np
import pandas as pd
from src.consts import PROMOTER_LENGTH
from src.utils import get_current_file_parent_path
from pathlib import Path
from scipy.stats import linregress

RNA_TYPE = 'p'
RAW_PCN_COLUMN_NAME = 'Raw Copy Number'
TARGET_PCN_COLUMN_NAME = 'pcn'
DATA_PATH = Path(get_current_file_parent_path(__file__).parent.parent, 'data')
DATA_FILE_PATH = Path(DATA_PATH, f'RNA{RNA_TYPE}_with_Raw_PCN.csv')
ADDITIONAL_DATA_FILE_PATH = Path(DATA_PATH, 'biology_results', f'PCN RNA{RNA_TYPE} results.xlsx')


def is_promoter_sequence_valid(sequence):
    return '-' not in sequence and len(sequence) == PROMOTER_LENGTH


def join_data_with_article_sequences(article_data, additional_data, additional_promoter_col='promoter seq'):
    merged_data = pd.merge(article_data, additional_data, how='inner', left_on='Promoter Sequence (-35 to +1)',
                           right_on=additional_promoter_col)
    merged_data.rename(columns={additional_data.columns.values[-1]: TARGET_PCN_COLUMN_NAME}, inplace=True)
    return merged_data


def get_measured_pcn():
    article_data = pd.read_csv(DATA_FILE_PATH, index_col=0)
    article_ddpcr_data = pd.read_excel(ADDITIONAL_DATA_FILE_PATH, sheet_name=f'ddPCR - RNA{RNA_TYPE}')[
        ['promoter seq', 'use for fit', 'ddpcr']]
    our_bio_data = pd.read_excel(ADDITIONAL_DATA_FILE_PATH, sheet_name=f'Our qPCR - RNA{RNA_TYPE}')[
        [f'RNA{RNA_TYPE} promoter', 'use for fit', 'PCN measured in Wet lab']]

    # filter invalid promoters (due to indel)
    our_bio_data = our_bio_data.loc[our_bio_data[f"RNA{RNA_TYPE} promoter"].apply(is_promoter_sequence_valid)]

    concat_ddpcr = join_data_with_article_sequences(article_data, article_ddpcr_data, 'promoter seq')
    concat_qpcr = join_data_with_article_sequences(article_data, our_bio_data, f'RNA{RNA_TYPE} promoter')

    raw_target_pcn_df = pd.concat((
        pd.DataFrame(
            {'raw_pcn': concat_ddpcr[RAW_PCN_COLUMN_NAME], 'target_pcn': concat_ddpcr[TARGET_PCN_COLUMN_NAME],
             'use_for_fit': concat_ddpcr['use for fit'], 'type': 'ddPCR'}),
        pd.DataFrame(
            {'raw_pcn': concat_qpcr[RAW_PCN_COLUMN_NAME], 'target_pcn': concat_qpcr[TARGET_PCN_COLUMN_NAME],
             'use_for_fit': concat_qpcr['use for fit'], 'type': 'qPCR'})),
        axis=0)

    return raw_target_pcn_df


def filter_measurements_fot_fit(measurements_df):
    return measurements_df.query('use_for_fit == "V"')


def get_log_log_linear_regression(raw_pcn, target_pcn):
    log_raw_pcn = np.log(raw_pcn)
    log_target_pcn = np.log(target_pcn)

    slope, intercept, _, _, _ = linregress(log_raw_pcn, log_target_pcn)

    return slope, intercept


def apply_linear_fit(slope, intercept):
    return lambda x: np.exp(slope * np.log(x) + intercept)


def custom_fit_and_transform_raw_pcn(raw_pcn):
    measured_pcn = get_measured_pcn()
    measured_pcn_for_fit = filter_measurements_fot_fit(measured_pcn)
    slope, intercept = get_log_log_linear_regression(measured_pcn_for_fit['raw_pcn'], measured_pcn_for_fit['target_pcn'])
    return raw_pcn.apply(apply_linear_fit(slope, intercept))
