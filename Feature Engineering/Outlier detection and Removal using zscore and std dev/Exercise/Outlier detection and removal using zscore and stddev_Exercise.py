#You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,
#(1) Remove outliers using percentile technique first. Use [0.001, 0.999] for lower and upper bound percentiles
#(2) After removing outliers in step 1, you get a new dataframe.
#(3) On step(2) dataframe, use 4 standard deviation to remove outliers
#(4) Plot histogram for new dataframe that is generated after step (3). Also plot bell curve on same histogram
#(5) On step(2) dataframe, use zscore of 4 to remove outliers. This is quite similar to step (3) and you will get exact same result


import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Feature Engineering\Outlier detection and Removal using zscore and std dev\Exercise\bhp.csv')
print(df)

print(df.price_per_sqft.describe())

# Plot histogram of price_per_sqft
plt.hist(df.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('price_per_sqft')
plt.ylabel('Count')
plt.show()

# Log scale can make the visualization better
plt.hist(df.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('Price per square ft')
plt.ylabel('Count')
plt.yscale('log')
plt.show()

# Remove outliers using percentile technique
min_threshold, max_threshold = df.price_per_sqft.quantile([0.001, 0.999])
print(min_threshold, max_threshold)

df2 = df[(df.price_per_sqft < max_threshold) & (df.price_per_sqft > min_threshold)]
print(df2)
print(df2.price_per_sqft.describe())

# Remove outliers using 4 standard deviations
mean = df2.price_per_sqft.mean()
std = df2.price_per_sqft.std()
upper_limit = mean + 4 * std
lower_limit = mean - 4 * std

final = df2[(df2.price_per_sqft < upper_limit) & (df2.price_per_sqft > lower_limit)]
print(final)

# Plot histogram for new dataframe after removing outliers using 4 standard deviations
matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.hist(final.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('price_per_sqft')
plt.ylabel('Count')
plt.show()

# Log scale can make the visualization better
plt.hist(final.price_per_sqft, bins=20, rwidth=0.8)
plt.xlabel('Price per square ft')
plt.ylabel('Count')
plt.yscale('log')
plt.show()

# Plot bell curve on the same histogram
rng = np.arange(final.price_per_sqft.min(), final.price_per_sqft.max(), 0.1)
plt.plot(rng, norm.pdf(rng, final.price_per_sqft.mean(), final.price_per_sqft.std()))
plt.show()

# Remove outliers using z-score of 4
df2['zscore'] = (df2.price_per_sqft - df2.price_per_sqft.mean()) / df2.price_per_sqft.std()
outliers_zscore = df2[(df2.zscore < -4) | (df2.zscore > 4)]
final_zscore = df2[(df2.zscore > -4) & (df2.zscore < 4)]
print(final_zscore)
