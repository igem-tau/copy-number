from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def run_lasso(X_train, X_test, y_train, y_test, data_title: str = None):
    # define the model
    lasso_model = Lasso(alpha=0.3, max_iter=5000)
    # training the model
    _ = lasso_model.fit(X_train, y_train)
    # Predict
    y_pred = lasso_model.predict(X_test)

    # evaluation
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for lasso: {r2}")

    # feature importance
    importance = lasso_model.coef_
    features = X_train.columns[importance != 0]
    importance = importance[importance != 0]
    print(f'number of non-zero coefficients is: {len(features)}')

    # plot best 20 features
    idx = (-importance).argsort()[:20]
    coefs = pd.DataFrame(
        importance[idx],
        columns=['Coefficients'], index=features[idx]
    )

    coefs.plot(kind='barh', figsize=(9, 7))
    plt.title(f'Lasso model for {data_title}')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.show()
