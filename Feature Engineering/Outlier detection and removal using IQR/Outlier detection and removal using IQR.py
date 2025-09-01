import pandas as pd
df = pd.read_csv('heights.csv')
print(df)

print(df.describe())

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
