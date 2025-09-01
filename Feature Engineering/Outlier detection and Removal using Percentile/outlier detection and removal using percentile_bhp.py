import pandas as pd
df = pd.read_csv('bhp.csv')
print(df)

print(df.describe())

min_threshold, max_threshold = df.price_per_sqft.quantile([0.001, 0.999])
print(min_threshold, max_threshold)

print(df[df.price_per_sqft < min_threshold])
print(df[df.price_per_sqft > max_threshold])

df2 = df[(df.price_per_sqft < max_threshold) & (df.price_per_sqft > min_threshold)]
print(df2)

print(df2.sample(10))
