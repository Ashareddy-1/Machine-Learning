import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

#Read from csv file using pandas
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Linear Regression single variable\Exercise\canada_per_capita_income.csv')
print(df)

#Now draw a scatter plt for this data
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='*')
plt.show()

#Train the model
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df[['per capita income (US$)']])

#Print the coefficients
print(reg.coef_[0])
print(reg.intercept_)

#Now plot the linear line
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='*')
plt.plot(df['year'], reg.predict(df[['year']]), color='blue')
plt.show()

#Now lets predict per capita income for year 2020
year_df = pd.DataFrame({'year': [2020]})
per_capita_income_prediction = reg.predict(year_df)
print(per_capita_income_prediction)
