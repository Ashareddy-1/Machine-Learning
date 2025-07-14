#Use this air bnb new york city data set and remove outliers using percentile based on price per night for a given apartment/home.
#You can use suitable upper and lower limits on percentile based on your intuition.
#Your goal is to come up with new pandas dataframe that doesn't have the outliers present in it.

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Feature Engineering\Outlier detection and Removal using Percentile\Exercise\AB_NYC_2019.csv')
print(df)

print(df.price.describe())

min_threshold, max_threshold = df.price.quantile([0.01, 0.999])
print(min_threshold, max_threshold)

print(df[df.price < min_threshold])
print(df[df.price > max_threshold])

df2 = df[(df.price < max_threshold) & (df.price > min_threshold)]
print(df2)
print(df2.price.describe())

print(df2.sample(10))
