from itertools import combinations
import pandas as pd
from scipy.special import comb
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from src.utils import get_current_file_parent_path

ROOT_PATH = get_current_file_parent_path(__file__).parent.parent
MEASUREMENTS_FILE = f'{ROOT_PATH}/data/biology_results/PCN RNAp results.xlsx'
PREDICTIONS_FILE = f'{ROOT_PATH}/copy_num_predictions_RNAp_voting.csv'
measurements = pd.read_excel(MEASUREMENTS_FILE, sheet_name='Our qPCR - RNAp')
predictions = pd.read_csv(PREDICTIONS_FILE)

merged = pd.merge(predictions, measurements, left_on='Promoter Sequence (-35 to +1)', right_on='RNAp promoter')
merged = merged[['Variant name', 'Promoter Sequence (-35 to +1)', 'Copy Number', 'PCN measured in Wet lab']]
merged = merged.rename(columns={'Copy Number': 'prediction', 'PCN measured in Wet lab': 'measured'}).reset_index(
    drop=True)

num_records = merged.shape[0]
correlations = {
    'num_records': [],
    'selected': [],
    'spearman': [],
    'spearman_pv': [],
    'pearson': [],
    'pearson_pv': []
}
for num_records_selected in range(5, num_records + 1):
    num_combinations = comb(num_records, num_records_selected, True)
    for combination in tqdm(combinations(range(num_records), num_records_selected), total=num_combinations,
                            desc=f'calculatin all options of length: {num_records_selected}'):
        spearman_corr = spearmanr(merged.loc[list(combination), 'prediction'].values.tolist(),
                                  merged.loc[list(combination), 'measured'].values.tolist())
        pearson_corr = pearsonr(merged.loc[list(combination), 'prediction'],
                                merged.loc[list(combination), 'measured'])

        correlations['num_records'].append(num_records_selected),
        correlations['selected'].append(', '.join(merged.loc[list(combination), 'Variant name'].values)),
        correlations['spearman'].append(spearman_corr.statistic),
        correlations['spearman_pv'].append(spearman_corr.pvalue),
        correlations['pearson'].append(pearson_corr.statistic),
        correlations['pearson_pv'].append(pearson_corr.pvalue)

correlations = pd.DataFrame(correlations).sort_values(by=['spearman', 'pearson'], ascending=False)
correlations.to_csv(f'{ROOT_PATH}/data/prediction_correlations_to_biology_measurements.csv', index=False)
