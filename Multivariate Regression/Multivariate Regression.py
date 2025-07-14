import pandas as pd
import numpy as np
import math
from sklearn import linear_model

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\Machine Learning\Multivariate Regression\homeprices.csv')
print(df)

median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)

table = df.bedrooms.fillna(median_bedrooms)
print(table)

df.bedrooms = table
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df['price'])
print(reg)

print(reg.coef_)
print(reg.intercept_)


new_data = pd.DataFrame({'area': [3000], 'bedrooms': [3], 'age': [40]})
price_prediction = reg.predict(new_data)
print(price_prediction)

