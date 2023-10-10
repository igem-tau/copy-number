import numpy as np
import pandas as pd
from src.consts import PROMOTER_LENGTH, RNA_DATA_COLUMNS
from src.data_prep.raw_pcn_fitting import get_measured_pcn, get_log_log_linear_regression
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
    return '-' not in sequence and len(sequence) == PROMOTER_LENGTH


def join_data_with_predicted_sequences(predicted_data, measurement_data, measurement_promoter_col='promoter seq'):
    merged_data = pd.merge(predicted_data, measurement_data, how='inner', left_on='Promoter Sequence (-35 to +1)',
                           right_on=measurement_promoter_col)
    merged_data.rename(columns={measurement_data.columns.values[-1]: TARGET_PCN_COLUMN_NAME}, inplace=True)
    return merged_data


def article_fit_func(raw_pcn):
    return 123 * raw_pcn ** 0.37 - 28


def scatter_plot_raw_target(raw_traget_pcn_df):
    fig = px.scatter(raw_traget_pcn_df, x='raw_pcn', y='target_pcn', color='type',
                     labels={'raw_pcn': 'log raw copy number', 'target_pcn': 'log ddPCR/qPCR'})
    fig.update_traces(marker_size=5)
    return fig


def permutations_p_value(prediction, target, out_of=100, corr='spearman'):
    if corr == 'spearman':
        corr_method = spearmanr
    elif corr == 'pearson':
        corr_method = pearsonr
    else:
        raise ValueError(
            'transformation_to_biology_results.permutations_p_value: corr supports only: "spearman" or "pearson"')
    base_score = corr_method(prediction, target).statistic
    better_count = 0
    permutations = []
    while len(permutations) < out_of:
        new_permutation = np.random.permutation(prediction)
        if all([(new_permutation != p).any() for p in permutations]):
            permutations.append(new_permutation)
            permutation_score = corr_method(new_permutation, target).statistic
            if permutation_score >= base_score:
                better_count += 1

    return max(1, better_count) / out_of


def pre_model_prep():
    measured_pcn_for_fit = get_measured_pcn(for_fit=True)

    measured_pcn_for_fit['raw_pcn'] = measured_pcn_for_fit['raw_pcn'].apply(np.log)
    measured_pcn_for_fit['target_pcn'] = measured_pcn_for_fit['target_pcn'].apply(np.log)

    fig = scatter_plot_raw_target(measured_pcn_for_fit)

    slope, intercept, pearson_corr, p_value, std_err = linregress(measured_pcn_for_fit['raw_pcn'],
                                                                  measured_pcn_for_fit['target_pcn'])
    print(f'{slope=} {intercept=} {pearson_corr=} {p_value=} {std_err=}')
    spearman_corr = spearmanr(measured_pcn_for_fit['raw_pcn'], measured_pcn_for_fit['target_pcn'])
    print('spearman', spearman_corr)

    x_space = pd.Series(np.linspace(measured_pcn_for_fit['raw_pcn'].min(), measured_pcn_for_fit['raw_pcn'].max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: slope * x + intercept)), name='linear regression'))

    # fig.add_trace(go.Scatter(x=x_space,
    #                          y=(x_space.apply(lambda x: np.exp(slope * x + intercept))),
    #                          name='pcn by linear regression'))

    fig.update_layout(title=dict(text=f'raw copy number power log fit, y={slope:.3f}x + {intercept:.3f}'))
    fig.add_trace(go.Scatter(
        x=[measured_pcn_for_fit['raw_pcn'].min(), measured_pcn_for_fit['raw_pcn'].min()],
        y=[measured_pcn_for_fit['target_pcn'].max(), measured_pcn_for_fit['target_pcn'].max() - .3],
        mode="text",
        name='correlations metrics',
        text=[f'spearman: {spearman_corr.statistic:.3f}, p-value: {spearman_corr.pvalue:.3f}',
              f'pearson: {pearson_corr:.3f}, p-value: {p_value:.3f}'],
        textposition="top right"
    ))

    plotly.offline.plot(fig, filename=str(Path(DATA_PATH, 'figures', 'raw_pcn_fit_search.html')))


def post_model_estimation():
    measurements_predictions = get_measured_pcn(with_duplicates=False, matching='predictions')
    measurements_predictions_for_val = measurements_predictions.query('use_for_fit == "X"').copy()

    measurements_predictions_for_val['raw_pcn'] = measurements_predictions_for_val['raw_pcn'].apply(np.log)
    measurements_predictions_for_val['target_pcn'] = measurements_predictions_for_val['target_pcn'].apply(np.log)

    slope, intercept, pearson_corr, p_value, std_err = linregress(
        measurements_predictions_for_val['raw_pcn'],
        measurements_predictions_for_val['target_pcn'])

    print(f'{slope=} {intercept=} {pearson_corr=} {p_value=} {std_err=}')
    spearman_corr = spearmanr(measurements_predictions_for_val['raw_pcn'],
                              measurements_predictions_for_val['target_pcn'])
    print('spearman', spearman_corr)

    spearman_permutations_p_value = permutations_p_value(measurements_predictions_for_val['raw_pcn'],
                                                         measurements_predictions_for_val['target_pcn'], out_of=100)
    print(f'{spearman_permutations_p_value=}, out of 100')
    spearman_permutations_p_value = permutations_p_value(measurements_predictions_for_val['raw_pcn'],
                                                         measurements_predictions_for_val['target_pcn'], out_of=1000)
    print(f'{spearman_permutations_p_value=}, out of 1000')

    pearson_permutations_p_value = permutations_p_value(measurements_predictions_for_val['raw_pcn'],
                                                        measurements_predictions_for_val['target_pcn'], corr='pearson',
                                                        out_of=100)
    print(f'{pearson_permutations_p_value=}, out of 100')
    pearson_permutations_p_value = permutations_p_value(measurements_predictions_for_val['raw_pcn'],
                                                        measurements_predictions_for_val['target_pcn'], corr='pearson',
                                                        out_of=1000)
    print(f'{pearson_permutations_p_value=}, out of 1000')

    fig = px.scatter(measurements_predictions_for_val, x='raw_pcn', y='target_pcn', color='type',
                     labels={'raw_pcn': 'log predicted copy number', 'target_pcn': 'log ddPCR/qPCR'})
    fig.update_traces(marker_size=5)
    x_space = pd.Series(np.linspace(measurements_predictions_for_val['raw_pcn'].min(),
                                    measurements_predictions_for_val['raw_pcn'].max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: slope * x + intercept)), name='linear regression'))

    # fig.add_trace(go.Scatter(x=x_space,
    #                          y=(x_space.apply(lambda x: np.exp(slope * x + intercept))),
    #                          name='pcn by linear regression'))

    fig.update_layout(
        title=dict(text=f'predicted copy number power log validation fit, y={slope:.3f}x + {intercept:.3f}'))
    fig.add_trace(go.Scatter(
        x=[measurements_predictions_for_val['raw_pcn'].min(),
           measurements_predictions_for_val['raw_pcn'].min()],
        y=[measurements_predictions_for_val['target_pcn'].max(),
           measurements_predictions_for_val['target_pcn'].max() - .3],
        mode="text",
        name='correlations metrics',
        text=[f'spearman: {spearman_corr.statistic:.3f}, p-value: {spearman_corr.pvalue:.3f}',
              f'pearson: {pearson_corr:.3f}, p-value: {p_value:.3f}'],
        textposition="top right"
    ))

    plotly.offline.plot(fig, filename=str(Path(DATA_PATH, 'figures', 'predicted_pcn_fit_validation.html')))


if __name__ == '__main__':
    pre_model_prep()
    print()
    post_model_estimation()
