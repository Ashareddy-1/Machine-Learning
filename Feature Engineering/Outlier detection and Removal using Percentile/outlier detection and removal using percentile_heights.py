import pandas as pd
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Feature Engineering\Outlier detection and Removal using Percentile\heights.csv')
print(df)

max_threshold = df['height'].quantile(0.95)
print(max_threshold)

max_outlier = df[df['height'] > max_threshold]
print(max_outlier)

min_threshold = df['height'].quantile(0.05)
print(min_threshold)
                 
min_outlier = df[df['height'] < min_threshold]
print(min_outlier)

final = df[(df['height']<max_threshold) & (df['height']>min_threshold)]
print(final)
