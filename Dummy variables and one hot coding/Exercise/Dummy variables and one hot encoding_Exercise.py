#predict price of a mercedez benz that is 4 yr old with mileage 45000
#Predict price of a BMW X5 that is 7yr old with mileage 86000
#Find the score of model.

#For any question first plot the data and check which regression works based on plot.
#As Car Model is in text format so using dummy variables

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

#Read from csv file using pandas
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Dummy variables and one hot coding\Exercise\carprices.csv')
print(df)

#Dummies
dummies = pd.get_dummies(df.Car_Model)
print(dummies)

merged = pd.concat([df, dummies], axis = 'columns')
print(merged)

final = merged.drop(['Car_Model'], axis = 'columns')
print(final)


model = LinearRegression()

x = final.drop('Sell_Price', axis = 'columns')
print(x)
y = final.Sell_Price
print(y)

training = model.fit(x,y)
print(training)

#predict price of a mercedez benz that is 4 yr old with mileage 45000
new_data = pd.DataFrame({'Mileage': [45000], 'Age(yrs)': [4], 'Audi A5': [False], 'BMW X5': [False], 'Mercedez Benz C class': [True]})
prediction = model.predict(new_data)
print(prediction)

#Predict price of a BMW X5 that is 7yr old with mileage 86000
new_data = pd.DataFrame({'Mileage': [86000], 'Age(yrs)': [7], 'Audi A5': [False], 'BMW X5': [True], 'Mercedez Benz C class': [False]})
prediction = model.predict(new_data)
print(prediction)

#Find the score of model.
score = model.score(x,y)
print(score)
