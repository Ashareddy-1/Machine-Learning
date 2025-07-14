#You are given height_weight.csv file which contains heights and weights of 1000 people. Dataset is taken from here, https://www.kaggle.com/mustafaali96/weight-height
#You need to do this,
#(1) Load this csv in pandas dataframe and first plot histograms for height and weight parameters
#(2) Using IQR detect weight outliers and print them
#(3) Using IQR, detect height outliers and print them

import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Feature Engineering\Outlier detection and removal using IQR\Exercise\height_weight.csv')
print(df)

print(df.describe())

#Outliers Height

Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)

print(Q1, Q3)

IQR = Q3-Q1
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

print(lower_limit, upper_limit)

outliers = df[(df.height<lower_limit) | (df.height>upper_limit)]
print(outliers)
final = df[(df.height>lower_limit) & (df.height<upper_limit)]
print(final)

#Outliers weight

Q1 = df.weight.quantile(0.25)
Q3 = df.weight.quantile(0.75)

print(Q1, Q3)

IQR = Q3-Q1
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

print(lower_limit, upper_limit)

outliers = df[(df.weight<lower_limit) | (df.weight>upper_limit)]
print(outliers)
final = df[(df.weight>lower_limit) & (df.weight<upper_limit)]
print(final)
