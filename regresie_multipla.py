import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Dataset.csv")

X = df.drop(['PRICE'], axis=1)
X = pd.get_dummies(X)
y = np.log(df['PRICE'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

def plot_train_predictions():
    y_pred = model.predict(X_train)

    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_train, alpha=0.5)
    ax.set_xlabel('Predicted Sale Price')
    ax.set_ylabel('Actual Sale Price')
    ax.set_title('Comparing Predicted and Actual Sale Prices (Train)')

    max_val = max(y_pred.max(), y_train.max())
    min_val = min(y_pred.min(), y_train.min())
    ax.plot([min_val, max_val], [min_val, max_val], color="red")
    return fig

def plot_test_predictions():
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, alpha=0.5)
    ax.set_xlabel('Predicted Sale Price')
    ax.set_ylabel('Actual Sale Price')
    ax.set_title('Comparing Predicted and Actual Sale Prices (Test)')

    max_val = max(y_pred.max(), y_test.max())
    min_val = min(y_pred.min(), y_test.min())
    ax.plot([min_val, max_val], [min_val, max_val], color="red")
    return fig

def show_metrics():
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def predict_single_house():
    single_pred = model.predict(X_test[2:3])
    return np.exp(single_pred)[0]
