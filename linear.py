import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error




#data reading
dataset = pd.read_csv("C:/Users/USER/Downloads/California_Houses.csv")
dataset.head()

#define x (independent var.) and y (dependent var.)
x = dataset.drop(['Median_House_Value'], axis=1).values
y=dataset['Median_House_Value'].values
print(x)
print(y)

#split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

#model training
ml=LinearRegression()
ml.fit(x_train, y_train)
# Lasso Regression
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
lasso_model.fit(x_train, y_train)
lasso_y_pred = lasso_model.predict(x_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_mae = mean_absolute_error(y_test, lasso_y_pred)
print("Lasso Regression MSE:", lasso_mse)
print("Lasso Regression MAE:", lasso_mae)

# Ridge Regression
ridge_model = Ridge(alpha=0.1)  # You can adjust the alpha parameter
ridge_model.fit(x_train, y_train)
ridge_y_pred = ridge_model.predict(x_test)
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_mae = mean_absolute_error(y_test, ridge_y_pred)
print("Ridge Regression MSE:", ridge_mse)
print("Ridge Regression MAE:", ridge_mae)

#predicting test set results
y_pred = ml.predict(x_test)
print(y_pred)

#model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculating Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)


#plotting the model
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('actual')
plt.ylabel('predicted')