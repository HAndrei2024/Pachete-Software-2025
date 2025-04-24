import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score


df = pd.read_csv("Dataset.csv")

#X = df[['BEDS', 'BATH', 'MetriPatratiLocuinta']]
X = df.drop(['PRICE'], axis=1)
X = pd.get_dummies(X)
#y = df['PRICE']
y = np.log(df['PRICE'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=63, test_size=.20)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
y_predicted = model.predict(X_train)

plt.scatter(y_predicted, y_train)
plt.xlabel('Predicted Sale Price')
plt.ylabel('Actual Sale Price')
plt.title('Comparing Predicted and Actual Sale Prices')
max_val = max(y_predicted.max(), y_test.max())
min_val = min(y_predicted.min(), y_test.min())
plt.plot([min_val, max_val], [min_val, max_val], color="red")
plt.show()

print('RMSE LR Train set: ', mean_squared_error(y_train, y_predicted))


y_predicted = model.predict(X_test)

plt.scatter(y_predicted, y_test)
plt.xlabel('Predicted Sale Price')
plt.ylabel('Actual Sale Price')
plt.title('Comparing Predicted and Actual Sale Prices')
max_val = max(y_predicted.max(), y_test.max())
min_val = min(y_predicted.min(), y_test.min())
plt.plot([min_val, max_val], [min_val, max_val], color="red")
plt.show()

print('MSE LR test set: ', mean_squared_error(y_test, y_predicted))

# Media pătratelor diferențelor dintre valori reale și prezise.
mse = mean_squared_error(y_test, y_predicted)
print(f"MSE: {mse:.4f}")

# Rădăcina pătrată a MSE. Este în aceleași unități ca ținta (SalePrice în log).
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Media valorilor absolute ale erorilor.
mae = mean_absolute_error(y_test, y_predicted)
print(f"MAE: {mae:.4f}")

r2 = r2_score(y_test, y_predicted)
print(f"R² (R squared): {r2:.4f}")

single_predicted = model.predict(X_test[2:3])
print(single_predicted)

print('Single House price prediction with LR:', np.exp(single_predicted))