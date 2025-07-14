# Do some exploratory data analysis to figure out which variables have direct and clear impact on employess retention(i.e; whether they leave the company or continue to work
# Plot bar charts showing impact of employees salaries on retention
#Plot bar charts showing corelation between department and employee retention
#Now build logistic regression model using thta were narrowed down in step 1
# Measure the accuracy of the model

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the data
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Logistic Regression\Exercise\HR_comma_sep.csv')
print(df)

# Separate the data based on the 'left' column
left = df[df.left == 1]
print(left.shape)

retained = df[df.left == 0]
print(retained.shape)

# Exclude non-numeric columns before applying groupby.mean()
numeric_df = df.select_dtypes(include=[np.number])
average = numeric_df.groupby('left').mean()
print(average)

# Plot salary vs left
pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.show()

# Plot department vs left
pd.crosstab(df.Department, df.left).plot(kind='bar')
plt.show()

# Create a subset of the DataFrame
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
print(subdf)

# Create dummy variables for the 'salary' column
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
print(df_with_dummies)

# Drop the original 'salary' column
df_with_dummies.drop('salary', axis='columns', inplace=True)
print(df_with_dummies)

# Define features and target variable
X = df_with_dummies
print(X)

y = df.left
print(y)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

# Train the logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
model.predict(X_test)
print(model.score(X_test, y_test))

