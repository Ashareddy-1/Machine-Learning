import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Naive Bayes Classifier\spam.csv')
print(df)

describe = df.groupby('Category').describe()
print(describe)

df['spam'] = df['Category'].apply(lambda x:1 if x=='spam' else 0)
print(df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()[:3]
print(x_train_count)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count, y_train)

emails = [
    'Hey Asha, can we get together to watch basketball game tomorrow?',
    'Upto 20% discount on parking, exclusively offer just for ou. Dont miss this reward!'
    ]
emails_count = v.transform(emails)
predict = model.predict(emails_count)
print(predict)

x_test_count = v.transform(x_test)
score = model.score(x_test_count, y_test)
print(score)

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(x_train, y_train)
clf.score(x_test, y_test)
clf.predict(emails)
