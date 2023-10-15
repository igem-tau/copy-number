import numpy as np
import pandas as pd
from src.consts import PROMOTER_LENGTH
from src.data_prep.raw_pcn_fitting import get_measured_pcn
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
DATA_FILE_PATH = Path(DATA_PATH, f'RNA{RNA_TYPE[0]}_with_Raw_PCN.csv')
ADDITIONAL_DATA_FILE_PATH = Path(DATA_PATH, 'biology_results', f'PCN RNA{RNA_TYPE[0]} results.xlsx')


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
                     labels={'raw_pcn': 'Log of Relative Copy Number', 'target_pcn': 'Log of ddPCR/qPCR',
                             'type': 'legend:'})
    fig.update_traces(marker_size=9)
    return fig


def permutations_p_value(prediction, target, out_of=100, corr='pearson'):
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
    pearson_permutations_p_value = permutations_p_value(
        measured_pcn_for_fit['raw_pcn'], measured_pcn_for_fit['target_pcn'],
        1000, 'pearson')
    print(f'{slope=} {intercept=} {pearson_corr=} {p_value=} {std_err=} {pearson_permutations_p_value=}')

    spearman_corr = spearmanr(measured_pcn_for_fit['raw_pcn'], measured_pcn_for_fit['target_pcn'])
    print('spearman', spearman_corr)

    x_space = pd.Series(np.linspace(measured_pcn_for_fit['raw_pcn'].min(), measured_pcn_for_fit['raw_pcn'].max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: slope * x + intercept)), name='linear fit'))

    fig.update_layout(title=dict(text=f'Relative Copy Number Power Log Fit, y={slope:.3f}x + {intercept:.3f}'))
    fig.add_trace(go.Scatter(
        x=[measured_pcn_for_fit['raw_pcn'].min()],
        y=[measured_pcn_for_fit['target_pcn'].max()],
        mode="text",
        name='correlations metrics',
        text=[f'Pearson: {pearson_corr:.3f}, p-value: {pearson_permutations_p_value:.3f}'],
        textposition="top right",
        showlegend=False
    ))

    plotly.offline.plot(fig, filename=str(Path(DATA_PATH, 'figures', 'raw_pcn_fit_search.html')))


def post_model_estimation(compare_to='validation'):
    measurements_predictions = get_measured_pcn(with_duplicates=False, matching='predictions')

    if compare_to == 'validation':
        measurements_predictions_for_val = measurements_predictions.query('use_for_fit == "X"').copy()
    elif compare_to == 'qpcr':
        measurements_predictions_for_val = measurements_predictions.query('type == "qPCR"').copy()
    else:
        raise ValueError('post_model_estimation: compare_to must be one of the following: "validation", "qpcr"')


    model_predictions = measurements_predictions_for_val['raw_pcn']
    biological_measurements = measurements_predictions_for_val['target_pcn']

    log_model_predictions = np.log(measurements_predictions_for_val['raw_pcn'])
    log_biological_measurements = np.log(measurements_predictions_for_val['target_pcn'])

    slope, intercept, pearson_corr, p_value, std_err = linregress(log_model_predictions, log_biological_measurements)

    pearson_permutations_p_value = permutations_p_value(log_model_predictions, log_biological_measurements, 1000,
                                                        'pearson')

    print(f'{slope=} {intercept=} {pearson_corr=} {p_value=} {std_err=} {pearson_permutations_p_value=}')
    spearman_corr = spearmanr(model_predictions, log_biological_measurements)
    print('spearman', spearman_corr)

    fig = px.scatter(pd.DataFrame(
        dict(predictions=model_predictions, target_pcn=biological_measurements, type=measurements_predictions['type'])),
        x='predictions', y='target_pcn', color='type',
        labels={'predictions': 'Predicted Copy Number', 'target_pcn': 'ddPCR/qPCR', 'type': 'legend:'})
    fig.update_traces(marker_size=9)

    x_space = pd.Series(np.linspace(model_predictions.min(), model_predictions.max(), 1000))
    fig.add_trace(go.Scatter(x=x_space,
                             y=(x_space.apply(lambda x: np.exp(slope * np.log(x) + intercept))),
                             name='linear fit'))

    fig.update_layout(
        title=dict(text=f'Predicted Copy Number Power Log Validation Fit, y={slope:.3f}x + {intercept:.3f}'),
        xaxis=dict(tickmode='linear', dtick=0.3, tickformat='.0f', type='log'),
        yaxis=dict(tickmode='linear', dtick=0.3, tickformat='.0f', type='log')
    )
    fig.add_trace(go.Scatter(
        x=[model_predictions.min()],
        y=[biological_measurements.max()],
        mode="text",
        name='correlations metrics',
        text=[f'Pearson: {pearson_corr:.3f}, p-value: {pearson_permutations_p_value:.3f}'],
        textposition="top right",
        showlegend=False
    ))

    plotly.offline.plot(fig, filename=str(Path(DATA_PATH, 'figures', 'predicted_pcn_fit_validation.html')))


if __name__ == '__main__':
    pre_model_prep()
    print()
    post_model_estimation(compare_to='qpcr')
