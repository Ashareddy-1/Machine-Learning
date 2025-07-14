# heart disease dataset heart.csv in Exercise folder and do following, (credits of dataset: https://www.kaggle.com/fedesoriano/heart-failure-prediction)
#Load heart disease dataset in pandas dataframe
#Remove outliers using Z score. Usual guideline is to remove anything that has Z score > 3 formula or Z score < -3
#Convert text columns to numbers using label encoding and one hot encoding
#Apply scaling
#Build a classification model using support vector machine. Use standalone model as well as Bagging model and check if you see any difference in the performance.
#Now use decision tree classifier. Use standalone model as well as Bagging and check if you notice any difference in performance
#Comparing performance of svm and decision tree classifier figure out where it makes most sense to use bagging and why. Use internet to figure out in what conditions bagging works the best.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Ensemble_learning\Exercise\heart.csv')

# Calculate Z-scores for numerical columns
z_scores = stats.zscore(df.select_dtypes(include=[np.number]))

# Create a DataFrame with Z-scores
z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=[np.number]).columns)

# Filter out rows with Z-score > 3 or Z-score < -3
df_filtered = df[(z_scores_df < 3).all(axis=1) & (z_scores_df > -3).all(axis=1)].copy()
print(df_filtered)

# Encode categorical variables using .loc to avoid SettingWithCopyWarning
le_Sex = LabelEncoder()
le_ChestPainType = LabelEncoder()
le_RestingECG = LabelEncoder()
le_ExerciseAngina = LabelEncoder()
le_ST_Slope = LabelEncoder()

df_filtered.loc[:, 'Sex'] = le_Sex.fit_transform(df_filtered['Sex'])
df_filtered.loc[:, 'ChestPainType'] = le_ChestPainType.fit_transform(df_filtered['ChestPainType'])
df_filtered.loc[:, 'RestingECG'] = le_RestingECG.fit_transform(df_filtered['RestingECG'])
df_filtered.loc[:, 'ExerciseAngina'] = le_ExerciseAngina.fit_transform(df_filtered['ExerciseAngina'])
df_filtered.loc[:, 'ST_Slope'] = le_ST_Slope.fit_transform(df_filtered['ST_Slope'])

print("Original DataFrame:")
print(df)
print("\nFiltered DataFrame:")
print(df_filtered)

# Ensure consistent numbers of samples in the input variables
x = df_filtered.drop('HeartDisease', axis='columns')
y = df_filtered['HeartDisease']

print(x)
print(y)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=30)

print("Training and testing sets created successfully.")

# Perform cross-validation with DecisionTreeClassifier
scores = cross_val_score(svm.SVC(), x_scaled, y, cv=5)
print(scores.mean())

# Train and evaluate BaggingClassifier with DecisionTreeClassifier as base estimator
bag_model = BaggingClassifier(
    estimator=svm.SVC(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(x_train, y_train)
print(bag_model.oob_score_)
score = bag_model.score(x_test, y_test)
print(scores.mean())

# Perform cross-validation with BaggingClassifier
bag_model = BaggingClassifier(
    estimator=svm.SVC(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, x_scaled, y, cv=5)
print(scores.mean())

# Perform cross-validation with DecisionTreeClassifier
scores = cross_val_score(svm.SVC(), x_scaled, y, cv=5)
print(scores.mean())

# Perform cross-validation with DecisionTreeClassifier
scores = cross_val_score(DecisionTreeClassifier(), x_scaled, y, cv=5)
print(scores.mean())

# Train and evaluate BaggingClassifier with DecisionTreeClassifier as base estimator
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(x_train, y_train)
print(bag_model.oob_score_)
score = bag_model.score(x_test, y_test)
print(scores.mean())
# Perform cross-validation with BaggingClassifier
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, x_scaled, y, cv=5)
print(scores.mean())
