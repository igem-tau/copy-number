import pandas as pd
from joblib import load


def remove_zero_variance_features(X: pd.DataFrame) -> pd.DataFrame:
  zero_variance_cols = X.columns[X.var() == 0]
  return X.drop(zero_variance_cols, axis=1)


def get_features_df():
    data = load('DataFrames_with_features.joblib')
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']
    return data
