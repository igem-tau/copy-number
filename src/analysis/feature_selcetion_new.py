from boruta import BorutaPy
from BorutaShap import BorutaShap
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from src.utils import get_current_file_parent_path
from xgboost import XGBRegressor

'''
https://github.com/Ekeany/Boruta-Shap
https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
'''

CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')

if __name__ == '__main__':
    # # load data
    # RNAp_data = get_RNAp_data()
    #
    # # split data to train and validation
    # RNAp_data, RNAp_data_val = split_for_validation(RNAp_data)
    #
    # # features extraction
    # data = {}
    # RNAp_X, RNAp_y = generate_features(RNAp_data, type='p', val=False)
    # data['RNAp_X'] = remove_zero_variance_features(RNAp_X)
    # data['RNAp_y'] = RNAp_y
    # pd.concat([data['RNAp_X'], data['RNAp_y']]).to_csv(os.path.join(DATA_PATH, 'p_RNA_DataFrames_with_features.csv'))

    data = load(Path(DATA_PATH, 'DataFrames_with_features.joblib'))
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']

    # feature vetting: select features based on correlations only
    # correlation between features and copy number (maximal) with MI
    mi = mutual_info_regression(RNAp_X, RNAp_y, discrete_features=(RNAp_X.dtypes == 'int64'))
    RNAp_X_new = RNAp_X.iloc[:, (mi>(mi.mean()))]
    # summary:
    print(f'Total number of features: {len(mi)}\nmedian: {np.median(mi)}\nmean: {mi.mean()}\nmax: {mi.max()}\nnumber of selected features (greater than the mean): {sum(mi>(mi.mean()))}')

    new_mi = mi[(mi>(mi.mean()))]

    # correlation between feature-feature (minimal) with MI
    new_features = pd.Series(RNAp_X_new.columns)
    corr_matrix = np.zeros((len(new_features), len(new_features)))
    for i in range(len(new_features)):
        for j in range(len(new_features)):
            if i == j:
                continue
            discrete_features_bool = True if RNAp_X_new[new_features[i]].dtype == 'int64' else False
            if RNAp_X_new[new_features[j]].dtype == 'int64':  # y is categorial
                corr_matrix[i, j] = mutual_info_classif(RNAp_X_new[new_features[i]].to_numpy().reshape(-1, 1), RNAp_X_new[new_features[j]],
                                                        discrete_features=discrete_features_bool)
            else:                                               # y is numerical
                corr_matrix[i, j] = mutual_info_regression(RNAp_X_new[new_features[i]].to_numpy().reshape(-1, 1), RNAp_X_new[new_features[j]],
                                                           discrete_features=discrete_features_bool)

    il1 = np.tril_indices(len(new_features))
    corr_matrix[il1] = np.nan
    print(f'median: {np.nanmedian(corr_matrix)}\nmean: {np.nanmean(corr_matrix)}\nmax: {np.nanmax(corr_matrix)}')

    (row, col) = (corr_matrix > np.nanquantile(corr_matrix, 0.99)).nonzero()  # TODO: think of different condition

    while len(row) > 0:
        values = np.array([row[0], col[0]])  # first pair
        inx = new_mi[values].argmin()  # find the feature with less correlation to the copy number
        new_features.drop(values[inx], inplace=True)  # erase from features Series

        # erase from row and col
        cur_inx_row = (row == values[inx]).nonzero()
        cur_inx_col = (col == values[inx]).nonzero()
        row = np.delete(row, np.concatenate((cur_inx_row, cur_inx_col), axis=1))
        col = np.delete(col, np.concatenate((cur_inx_row, cur_inx_col), axis=1))

    RNAp_X_new = RNAp_X_new.loc[:, new_features]

    # feature selection
    X_train, X_test, y_train, y_test = train_test_split(RNAp_X_new, RNAp_y, test_size=0.15) # TODO: stratify
    estimator = XGBRegressor(max_depth=3, n_estimators=10)  # XGBoost regressor with minimal properties (default is 6 and 100)
    # randomized the estimator parameters?
    selected_features_list = []
    for i in range(100):
        selector = SelectFromModel(estimator=estimator, threshold='0.05*mean').fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].to_list()
        selected_features_list += selected_features

    selected_features_final1 = set(selected_features_list)
    print(f'number of selected features with selector: {len(selected_features_final1)}\nthe selected features: {selected_features_final1}')

    # features selection option 2:
    estimator = XGBRegressor(max_depth=3, n_estimators=10)
    boruta = BorutaPy(estimator=estimator, n_estimators='auto', max_iter=100)
    boruta.fit(np.array(X_train), np.array(y_train))
    # Important features
    important = list(X_train.columns[boruta.support_])
    print(f'Features confirmed as important: {important}')
    # Tentative features
    tentative = list(X_train.columns[boruta.support_weak_])
    print(f'Unconfirmed features (tentative): {tentative}')
    print(f'number of selected features with boruta: {len(important)}')
    # Unimportant features
    # unimportant = list(X_train.columns[~(boruta.support_ | boruta.support_weak_)])
    # print(f'Features confirmed as unimportant: {unimportant}')

    # feature selection option 3:
    estimator = XGBRegressor(max_depth=3, n_estimators=10)

    Feature_Selector = BorutaShap(model=estimator,
                                  importance_measure='shap',
                                  classification=False)

    Feature_Selector.fit(X=RNAp_X_new, y=RNAp_y, n_trials=100, sample=False, #  TODO: sample_fraction=0.85,?
                         train_or_test='test', normalize=False,
                         verbose=True)

    # Feature_Selector.plot(which_features='all')
    features_to_remove = Feature_Selector.features_to_remove
    X_train_boruta_shap = X_train.drop(columns=features_to_remove)
    X_test_boruta_shap = X_test.drop(columns=features_to_remove)
    print(f'number of selected features with boruta shap: {len(X_train_boruta_shap.columns)}')
    print(f'selected features with boruta shap: {list(X_train_boruta_shap.columns)}')


    # evaluation of three methods:
    xgb_model = XGBRegressor()
    X_train_boruta = pd.concat([X_train.iloc[:, boruta.support_], X_train.iloc[:, boruta.support_weak_]], axis=1)
    X_test_boruta = pd.concat([X_test.iloc[:, boruta.support_], X_test.iloc[:, boruta.support_weak_]], axis=1)
    xgb_model.fit(X_train_boruta, y_train)
    y_pred_boruta = xgb_model.predict(X_test_boruta)

    xgb_model.fit(X_train_boruta_shap, y_train)
    y_pred_boruta_shap = xgb_model.predict(X_test_boruta_shap)

    xgb_model.fit(X_train.loc[:, selected_features_final1], y_train)
    y_pred_2 = xgb_model.predict(X_test.loc[:, selected_features_final1])

    r2_boruta = r2_score(y_test, y_pred_boruta)
    print(f'R^2 value for xgboost with boruta: {r2_boruta}')
    r2_boruta_shap = r2_score(y_test, y_pred_boruta_shap)
    print(f'R^2 value for xgboost with boruta shap: {r2_boruta_shap}')
    r2_selector = r2_score(y_test, y_pred_2)
    print(f'R^2 value for xgboost with selector: {r2_selector}')

    mse_score_boruta = mean_squared_error(y_test, y_pred_boruta)
    print('the mse score for xgboost %.5f with boruta' % mse_score_boruta)
    mse_score_boruta_shap = mean_squared_error(y_test, y_pred_boruta_shap)
    print('the mse score for xgboost %.5f with boruta shap' % mse_score_boruta_shap)
    mse_score_selector = mean_squared_error(y_test, y_pred_2)
    print('the mse score for xgboost %.5f with selector' % mse_score_selector)

    spearman_boruta, _ = spearmanr(y_test, y_pred_boruta)
    print(f'spearman correlation value for xgboost with boruta: {spearman_boruta}')
    spearman_boruta_shap, _ = spearmanr(y_test, y_pred_boruta_shap)
    print(f'spearman correlation value for xgboost with boruta shap: {spearman_boruta_shap}')
    spearman_selector, _ = spearmanr(y_test, y_pred_2)
    print(f'spearman correlation value for xgboost with selector: {spearman_selector}')
