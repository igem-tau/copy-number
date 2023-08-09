# Fixed Import with load Diabitis instead of load boston
from BorutaShap import BorutaShap
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectFromModel
from xgboost import XGBRegressor

'''
https://github.com/Ekeany/Boruta-Shap
https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
'''


def feature_selection(RNA_X, RNA_y):
    """
    Feature Selection for RNAp or RNAi.

    Accept: DataFrame of raining data and dataframe for test.
    Return: (Subset of data_train model with accepted features only, Array of accepted features, Array of Denied Features).

    Using Data vendding based on corrolation between features and droping uncorralated ones that are under the minimum.
    Using BorutaShap as model for feature selection (Wrapper Method).
    """

    # feature vetting: select features based on correlations only
    # correlation between features and copy number (maximal) with MI
    mi = mutual_info_regression(RNA_X, RNA_y, discrete_features=(RNA_X.dtypes == 'int64'))
    RNA_X_new = RNA_X.iloc[:, (mi > (mi.mean()))]
    new_mi = mi[(mi > (mi.mean()))]

    # correlation between feature-feature (minimal) with MI
    new_features = pd.Series(RNA_X_new.columns)
    corr_matrix = np.zeros((len(new_features), len(new_features)))
    for i in range(len(new_features)):
        for j in range(len(new_features)):
            if i == j:
                continue
            discrete_features_bool = True if RNA_X_new[new_features[i]].dtype == 'int64' else False
            if RNA_X_new[new_features[j]].dtype == 'int64':  # y is categorial
                corr_matrix[i, j] = mutual_info_classif(RNA_X_new[new_features[i]].to_numpy().reshape(-1, 1),
                                                        RNA_X_new[new_features[j]],
                                                        discrete_features=discrete_features_bool)
            else:  # y is numerical
                corr_matrix[i, j] = mutual_info_regression(RNA_X_new[new_features[i]].to_numpy().reshape(-1, 1),
                                                           RNA_X_new[new_features[j]],
                                                           discrete_features=discrete_features_bool)

    il1 = np.tril_indices(len(new_features))
    corr_matrix[il1] = np.nan

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

    RNA_X_new = RNA_X_new.loc[:, new_features]

    # feature selection - Boruta Sharp.
    estimator = XGBRegressor(max_depth=3,
                             n_estimators=10)  # XGBoost regressor with minimal properties (default is 6 and 100)
    Feature_Selector = BorutaShap(model=estimator,
                                  importance_measure='shap',
                                  classification=False)

    Feature_Selector.fit(X=RNA_X_new, y=RNA_y, n_trials=100, sample=False,  # TODO: sample_fraction=0.85,?
                         train_or_test='test', normalize=False,
                         verbose=True)

    # Return Values :
    features_to_remove = Feature_Selector.features_to_remove
    features_to_accept = Feature_Selector.accepted
    subset_of_data = Feature_Selector.Subset()

    return subset_of_data, features_to_accept, features_to_remove
