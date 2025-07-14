import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display the dataset and check for null values
print(df)
print(df.isnull().sum())
print(df.describe())
print(df.Outcome.value_counts())

# Separate features and target variable
X = df.drop("Outcome", axis="columns")
y = df.Outcome

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:3])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Perform cross-validation with DecisionTreeClassifier
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
print(scores)
print(scores.mean())

# Train and evaluate BaggingClassifier with DecisionTreeClassifier as base estimator
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
print(bag_model.oob_score_)
score = bag_model.score(X_test, y_test)
print(score)

# Perform cross-validation with BaggingClassifier
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=100, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, X, y, cv=5)
print(scores)
print
