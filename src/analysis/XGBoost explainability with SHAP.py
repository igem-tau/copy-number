from joblib import load
import pandas as pd
from pathlib import Path
import shap
from src.utils import get_current_file_parent_path
from xgboost import XGBRegressor


'''
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Front%20page%20example%20%28XGBoost%29.html
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
https://www.kaggle.com/code/bryanb/xgboost-explainability-with-shap
'''


CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')
data = load(Path(DATA_PATH, 'DataFrames_with_features.joblib'))

model = XGBRegressor()  # TODO: import the trained model + X_train
X_train = pd.DataFrame(None)
X_test = pd.DataFrame(None)

# Using a random sample of the dataframe for better time computation
X_sampled = X_train.sample(100, random_state=10)

# load JS visualization code to notebook
shap.initjs()

explainer = shap.TreeExplainer(model)

shap_values_train = explainer.shap_values(X_sampled)
shap_values_test = explainer.shap_values(X_test)

# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values_train, X_train)
shap.force_plot(explainer.expected_value, shap_values_test, X_test)

# summarize the effects of all the features
shap.summary_plot(shap_values_train, X_sampled)
shap.summary_plot(shap_values_test, X_test)

shap.summary_plot(shap_values_train, X_sampled, plot_type='bar')
shap.summary_plot(shap_values_test, X_test, plot_type='bar')
