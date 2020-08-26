#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#An all you need script to get scores from a model
file_path = "FuelConsumption.csv"
features = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
target = "CO2EMISSIONS"

def mean_absolute_percentage_error(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def load_csv_to_dataframe(file_path):
    if not os.path.exists(file_path):
        print("File does not exist")
    else:
        dataframe = pd.read_csv(file_path)
        print("Dataframe loaded")
        return dataframe

df = load_csv_to_dataframe(file_path) # Load data
print(f"Features: {features}")
print(f"Target: {target}")

X = df[features]
y = df[target].to_numpy().reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")
print(f"Residual sum of squares: {np.linalg.lstsq(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")
print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_pred)}")
# %%
