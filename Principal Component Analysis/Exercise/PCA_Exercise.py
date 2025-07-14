#Download heart disease dataset heart.csv in Exercise folder and do following, (credits of dataset: https://www.kaggle.com/fedesoriano/heart-failure-prediction)
#Load heart disease dataset in pandas dataframe
#Remove outliers using Z score. Usual guideline is to remove anything that has Z score > 3 formula or Z score < -3
#Convert text columns to numbers using label encoding and one hot encoding
#Apply scaling
#Build a classification model using various methods (SVM, logistic regression, random forest) and check which model gives you the best accuracy
#Now use PCA to reduce dimensions, retrain your model and see what impact it has on your model in terms of accuracy. Keep in mind that many times doing PCA reduces
#the accuracy but computation is much lighter and that's the trade off you need to consider while building models in real life.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
# Load the dataset
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Principal Component Analysis\Exercise\heart.csv')

# Calculate Z-scores for numerical columns
z_scores = stats.zscore(df.select_dtypes(include=[np.number]))

# Create a DataFrame with Z-scores
z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=[np.number]).columns)

# Filter out rows with Z-score > 3 or Z-score < -3
df_filtered = df[(z_scores_df < 3).all(axis=1) & (z_scores_df > -3).all(axis=1)].copy()

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

# Perform GridSearchCV on original scaled data
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_scaled, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_scores_original = pd.DataFrame(scores, columns=['model','best_score','best_params'])
print("GridSearchCV results on original scaled data:")
print(df_scores_original)

# Perform PCA on scaled data
pca = PCA(0.95)
x_pca = pca.fit_transform(x_scaled)
print(x_pca.shape)
print(x_pca)
print(pca.explained_variance_ratio_)
print(pca.n_components_)

# Perform GridSearchCV on PCA-transformed data
scores_pca = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_pca, y)
    scores_pca.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_scores_pca = pd.DataFrame(scores_pca, columns=['model','best_score','best_params'])
print("GridSearchCV results on PCA-transformed data:")
print(df_scores_pca)

