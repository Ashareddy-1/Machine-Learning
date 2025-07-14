import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\Machine Learning\Decision Tree\salaries.csv')
print(df)

inputs = df.drop('salary_more_then_100k', axis = 'columns')
print(inputs)
target = df['salary_more_then_100k']
print(target)

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
print(inputs)

inputs_n = inputs.drop(['company', 'job', 'degree'], axis = 'columns')
print(inputs_n)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

score = model.score(inputs_n, target)
print(score)

prediction = model.predict([[2,2,1]])
print(prediction)
