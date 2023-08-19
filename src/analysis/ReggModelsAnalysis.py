import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
from src.analysis import FeatureSelectionByModel
from src.utils import get_current_file_parent_path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xg
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from joblib import load



CURRENT_FOLDER_PATH = get_current_file_parent_path(__file__)
DATA_PATH = Path(CURRENT_FOLDER_PATH, '..', '..', 'data')

# Load the data features if exists, write if it doesn't
data = load(Path(DATA_PATH, 'DataFrames_with_features.joblib'))

# Extract X, Y, sequences - DataFrames

RNAp_stratify_train_val = data['RNAp_stratify_train_val']

RNAp_X_train_sequences = data['RNAp_X_train_sequences']
RNAp_X_train_features = data['RNAp_X_train']
RNAp_y_train = data['RNAp_y_train']

RNAp_X_val_sequences = data['RNAp_X_val_sequences']
RNAp_X_val_features = data['RNAp_X_val']
RNAp_y_val = data['RNAp_y_val']

RNAp_X_test_sequences = data['RNAp_X_test_sequences']
RNAp_X_test_features = data['RNAp_X_test']
RNAp_y_test = data['RNAp_y_test']
RNAp_stratify_test = data['RNAp_stratify_test']

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor(),
    'Support Vector Regression': SVR(),
    'K-Nearest Neighbors Regression': KNeighborsRegressor(),
    'Neural Network Regression': MLPRegressor(),
    'XGBoost Regression': xg.XGBRegressor(objective ='reg:linear', n_estimators = 6)
}

# Initialize lists to store model names and mean squared errors
model_names = []
mse_values = []
r2_scores = []
spearman_scores = []
Features_Selected_for_each_model = {}

for name, model in models.items():
    # Feature selection
    RNAp_selected_features_data = FeatureSelectionByModel.feature_selection(RNAp_X_train_features, RNAp_y_train,model)
    RNAp_selected_features = RNAp_selected_features_data['selected_features']
    print( name,"Features Selected :"+ str(RNAp_selected_features))

    # Data by selected features
    RNAp_FS_train = RNAp_X_train_features[RNAp_selected_features]
    RNAp_FS_val = RNAp_X_val_features[RNAp_selected_features]
    RNAp_FS_test = RNAp_X_test_features[RNAp_selected_features]
    for feature in RNAp_selected_features:
        if feature in Features_Selected_for_each_model:
            Features_Selected_for_each_model[feature].append(name)
        else:
            Features_Selected_for_each_model[feature] = [name]

    #Run Model
    RNAp_FS_train_val_X = pd.concat([RNAp_FS_train, RNAp_FS_val])
    RNAp_train_val_y = pd.concat([RNAp_y_train, RNAp_y_val])

    model.fit(RNAp_FS_train_val_X, RNAp_train_val_y)
    y_pred = model.predict(RNAp_FS_test)
    mse = mean_squared_error(RNAp_y_test, y_pred)
    r2 = r2_score(RNAp_y_test, y_pred)
    spearman_corr, _ = spearmanr(RNAp_y_test, y_pred)
    model_names.append(name)
    mse_values.append(mse)
    r2_scores.append(r2)
    spearman_scores.append(spearman_corr)

print(RNAp_selected_features, len(RNAp_selected_features))

for feature in Features_Selected_for_each_model.keys():
    print(feature,Features_Selected_for_each_model[feature])


# Plotting
x = np.arange(len(model_names))
# Plot MSE
plt.figure(figsize=(10, 4))
mse_bars = plt.bar(x, mse_values, align='center')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('Model Comparison: Mean Squared Error')
plt.xticks(x, model_names, rotation=45, ha='right')
for i, bar in enumerate(mse_bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{mse_values[i]:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Plot R2
plt.figure(figsize=(10, 4))
r2_bars =plt.bar(x, r2_scores, color='orange', align='center')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Model Comparison: R2 Score')
plt.xticks(x, model_names, rotation=45, ha='right')
for i, bar in enumerate(r2_bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{r2_scores[i]:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Plot Spearman Correlation
plt.figure(figsize=(10, 4))
spearman_bars = plt.bar(x, spearman_scores, color='green', align='center')
plt.xlabel('Models')
plt.ylabel('Spearman Correlation')
plt.title('Model Comparison: Spearman Correlation')
plt.xticks(x, model_names, rotation=45, ha='right')
for i, bar in enumerate(spearman_bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{spearman_scores[i]:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


## This is the winning regression which is random forest regression :
#TODO: Check for best Parameters for random dorest
estimator = RandomForestRegressor(n_estimators=30,max_depth = 5)
RNAp_selected_features_data = FeatureSelectionByModel.feature_selection(RNAp_X_train_features, RNAp_y_train, models['Random Forest Regression'])
RNAp_selected_features = RNAp_selected_features_data['selected_features']

# Data by selected features
RNAp_FS_train = RNAp_X_train_features[RNAp_selected_features]
RNAp_FS_val = RNAp_X_val_features[RNAp_selected_features]
RNAp_FS_test = RNAp_X_test_features[RNAp_selected_features]

# Run Model
RNAp_FS_train_val_X = pd.concat([RNAp_FS_train, RNAp_FS_val])
RNAp_train_val_y = pd.concat([RNAp_y_train, RNAp_y_val])

models['Random Forest Regression'].fit(RNAp_FS_train_val_X, RNAp_train_val_y)
y_pred = models['Random Forest Regression'].predict(RNAp_FS_test)

# Calculate metrics
mse = mean_squared_error(RNAp_y_test, y_pred)
r2 = r2_score(RNAp_y_test, y_pred)

# Create a scatter plot with regression line
plt.figure(figsize=(8, 6))
plt.scatter(RNAp_y_test, y_pred, color='blue', label='Actual Data')
plt.plot(RNAp_X_test_sequences['Promoter Sequence (-35 to +1)'], y_pred, color='red', label='Predicted Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Forest Regression')
plt.legend()
plt.tight_layout()

plt.show()

print(f"MSE: {mse:.3f}")
print(f"R2 Score: {r2:.3f}")

