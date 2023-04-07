from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def run_xgboost(X_train, X_test, y_train, y_test, importance_title: str = None):
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for xgboost: {r2}")

    plot_importance(xgb_model, max_num_features=20, title=importance_title)
    plt.show()
