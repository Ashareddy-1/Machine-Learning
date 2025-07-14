import pandas as pd
import numpy as np
import math
from word2number import w2n
from sklearn import linear_model

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Multivariate Regression\Exercise\hiring.csv')
print(df)

#To Fill 1st column in df which is experience blank fields to zero
table = df.iloc[:,0].fillna('zero') #df.ioc is used because if you not use this it will take in row not column
print(table)

#To cpnvert 1st column in df which is experience from words or strings to numbers
table_numbers = table.apply(w2n.word_to_num)
print(table_numbers)

#To replace 1st column in df with with table_numbers
df.iloc[:, 0] = table_numbers
print(df)

median_test_score = math.floor(df['test_score (out of 10)'].median())
print(median_test_score)

test_score_table = df['test_score (out of 10)'].fillna(median_test_score)
print(test_score_table)

df['test_score (out of 10)'] = test_score_table
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score (out of 10)', 'interview_score(out of 10)']], df['salary($)'])
print(reg)

print(reg.coef_)
print(reg.intercept_)


new_data = pd.DataFrame({'experience': [2], 'test_score (out of 10)': [9], 'interview_score(out of 10)': [6]})
salary_prediction = reg.predict(new_data)
print(salary_prediction)

new_data = pd.DataFrame({'experience': [12], 'test_score (out of 10)': [10], 'interview_score(out of 10)': [10]})
salary_prediction = reg.predict(new_data)
print(salary_prediction)
