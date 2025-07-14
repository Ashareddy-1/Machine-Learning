import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the data
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Linear Regression single variable\homeprices.csv')
print(df)

# Plot the data
plt.xlabel('area (sqr ft)')
plt.ylabel('price (US$)')
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.show()

# Train the model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

# Print the coefficients
print(f"Coefficient: {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

# Predict the price for a given area using a DataFrame
area_df = pd.DataFrame({'area': [3300]})
sqft_prediction = reg.predict(area_df)
print(f"Predicted price for 3300 sq ft: {sqft_prediction[0]}")

plt.xlabel('area', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.plot(df['area'],reg.predict(df[['area']]),color='blue')
plt.show()

d = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Linear Regression single variable\areas.csv')
print(d)
p = reg.predict(d)
d['prices'] = p
print(d)

d.to_csv("prediction.csv")

