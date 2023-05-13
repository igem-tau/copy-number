from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import os

FIGURES_PATH = os.path.join("..", "..", "..", "data", "copy_num", "figures")

def run_lasso(X_train, X_test, y_train, y_test, data_title: str = None, Best_param=None):
    if Best_param is not None:
        lasso_model = Lasso(**Best_param)
    else:
        lasso_model = Lasso(alpha=0.3, max_iter=5000)
    # training the model
    _ = lasso_model.fit(X_train, y_train)
    # Predict
    y_pred = lasso_model.predict(X_test)

    # evaluation
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 value for lasso: {r2}")
    mse_score = mean_squared_error(y_test, y_pred)
    print('the mse score for lasso %.5f' % mse_score)

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
    plt.savefig(os.path.join(FIGURES_PATH, f'Lasso feature importance {data_title}.jpg'))


    # evaluation plot
    f, ax = plt.subplots()
    plt.scatter(y_test, y_pred)
    plt.axline((0, 0), slope=1)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(0.8, 0.1, 'R2=%.4f' % r2, transform=ax.transAxes)
    plt.text(0.8, 0.2, 'MSE=%.4f' % mse_score, transform=ax.transAxes)
    plt.title(f'Lasso - {data_title}')
    plt.savefig(os.path.join(FIGURES_PATH, f'Lasso evaluation {data_title}.jpg'))
