#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error


# Sample data
data = {
    'X1': [60, 62, 67, 70, 71, 72, 75, 78],
    'X2': [22, 25, 24, 20, 15, 14, 14, 11],
    'X3': [15, 15, 14, 14, 13, 12, 12, 11],
    'X4': [2.5, 2.7, 2.9, 2.8, 3.5, 3.8, 3.9, 4.1],
    'X5': [6, 7.5, 9, 11, 14, 15.5, 18, 21],
    'Y': [140, 155, 159, 179, 192, 200, 212, 215]
}


df = pd.DataFrame(data)
x = df[["X1", "X2", "X3", "X4", "X5"]]
y = df["Y"]

# Define function to run regression models
#def run_regression_models(x, y, alpha=0.01):
    
      # Ridge model
    ridge_model = Ridge(alpha)
    ridge_model.fit(x, y)
    y_pred_ridge = ridge_model.predict(x)
    r2_ridge = r2_score(y, y_pred_ridge)
    mse_ridge = mean_squared_error(y, y_pred_ridge)
    print(f"Ridge Regression (alpha={alpha}):")
    print("R2 score:", r2_ridge)
    print("MSE:", mse_ridge)
    print("Coefficients:", ridge_model.coef_)
    print("Intercept:", ridge_model.intercept_)
    print(" ")
    
    
    
    
        # Lasso model
    lasso_model = Lasso(alpha)
    lasso_model.fit(x, y)
    y_pred_lasso = lasso_model.predict(x)
    r2_lasso = r2_score(y, y_pred_lasso)
    mse_lasso = mean_squared_error(y, y_pred_lasso)
    print(f"Lasso Regression (alpha={alpha}):")
    print("R2 score:", r2_lasso)
    print("MSE:", mse_lasso)
    print("Coefficients:", lasso_model.coef_)
    print("Intercept:", lasso_model.intercept_)
    print(" ")


# In[15]:


print('test')


# In[28]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Sample data
data = {
    'X1': [60, 62, 67, 70, 71, 72, 75, 78],
    'X2': [22, 25, 24, 20, 15, 14, 14, 11],
    'X3': [15, 15, 14, 14, 13, 12, 12, 11],
    'X4': [2.5, 2.7, 2.9, 2.8, 3.5, 3.8, 3.9, 4.1],
    'X5': [6, 7.5, 9, 11, 14, 15.5, 18, 21],
    'Y': [140, 155, 159, 179, 192, 200, 212, 215]
}

df = pd.DataFrame(data)
x = df[["X1", "X2", "X3", "X4", "X5"]]
y = df["Y"]
alpha = 0.1

# Ridge model
ridge_model = Ridge(alpha)
ridge_model.fit(x, y)
y_pred_ridge = ridge_model.predict(x)
r2_ridge = r2_score(y, y_pred_ridge)
mse_ridge = mean_squared_error(y, y_pred_ridge)
print("R2 score:", r2_ridge)
print("MSE:", mse_ridge)
print("Coefficients:", ridge_model.coef_)
print("Intercept:", ridge_model.intercept_)
print(" ")

# Lasso model
lasso_model = Lasso(alpha)
lasso_model.fit(x, y)
y_pred_lasso = lasso_model.predict(x)
r2_lasso = r2_score(y, y_pred_lasso)
mse_lasso = mean_squared_error(y, y_pred_lasso)
print("R2 score:", r2_lasso)
print("MSE:", mse_lasso)
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)
print(" ")


# In[33]:


df=pd.read_csv("Real Estate.csv")

# Display the count of remaining missing values in each column
missing_counts = df.isnull().sum()
print(missing_counts)
# Fill missing values in latitude and longitude with their respective column means
df["X5 latitude"] = df["X5 latitude"].fillna(df["X5 latitude"].mean())
df["X6 longitude"] = df["X6 longitude"].fillna(df["X6 longitude"].mean())

x=df.drop(columns=['Y house price of unit area'])
y=df['Y house price of unit area']

# Define function to run regression models
def run_regression_models(x, y, alpha):
    # Ridge model
    ridge_model = Ridge(alpha)
    ridge_model.fit(x, y)
    y_pred_ridge = ridge_model.predict(x)
    r2_ridge = r2_score(y, y_pred_ridge)
    mse_ridge = mean_squared_error(y, y_pred_ridge)
    print(f"\t alpha={alpha}:")
    print(f"Ridge Regression (alpha={alpha}):")
    print("R2 score:", r2_ridge)
    print("MSE:", mse_ridge)
    print("Coefficients:", ridge_model.coef_)
    print("Intercept:", ridge_model.intercept_)
    print(" ")

    # Lasso model
    lasso_model = Lasso(alpha)
    lasso_model.fit(x, y)
    y_pred_lasso = lasso_model.predict(x)
    r2_lasso = r2_score(y, y_pred_lasso)
    mse_lasso = mean_squared_error(y, y_pred_lasso)
    print(f"\t alpha={alpha}:")
    print(f"Lasso Regression (alpha={alpha}):")
    print("R2 score:", r2_lasso)
    print("MSE:", mse_lasso)
    print("Coefficients:", lasso_model.coef_)
    print("Intercept:", lasso_model.intercept_)
    print(" ")
# Run models for different alpha values
alpha_values = [0.01, 0.1, 1]
for alpha in alpha_values:
    run_regression_models(x, y, alpha)


# In[ ]:




