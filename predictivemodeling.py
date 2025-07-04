import pandas as pd
import numpy as np

data = pd.read_csv("Dataset .csv")

data['Name Length'] = data['Restaurant Name'].astype(str).apply(len)
data['Address Length'] = data['Address'].astype(str).apply(len)

data['Has Table Booking'] = data['Has Table booking'].astype(str).str.lower().map({'yes': 1, 'no': 0})
data['Has Online Delivery'] = data['Has Online delivery'].astype(str).str.lower().map({'yes': 1, 'no': 0})

data = data.dropna(subset=['Aggregate rating'])

X = data[['Name Length', 'Address Length', 'Has Table Booking', 'Has Online Delivery', 'Price range']].values
y = data['Aggregate rating'].values

n = len(X)
split = int(0.8 * n)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

weights = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

y_pred = X_test_b @ weights

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def R2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

print("\n📈 Predictive Modeling Results (Manual Linear Regression):")
print(f"Mean Absolute Error (MAE): {MAE(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {MSE(y_test, y_pred):.2f}")
print(f"R² Score: {R2(y_test, y_pred):.2f}")