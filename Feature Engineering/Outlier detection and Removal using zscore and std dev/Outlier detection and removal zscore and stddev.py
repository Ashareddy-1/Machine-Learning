import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

matplotlib.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('heights.csv')
print(df)

plt.hist(df.height, bins=20, rwidth=0.8)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
plt.show()

print(df.height.min())
print(df.height.max())
print(df.describe())

rng = np.arange(df.height.min(), df.height.max(), 0.1)
plt.plot(rng, norm.pdf(rng, df.height.mean(), df.height.std()))
plt.show()

mean = df.height.mean()
print(mean)

std = df.height.std()
print(std)

upper_limit = df.height.mean() + 3*df.height.std()
print(upper_limit)

lower_limit = df.height.mean() - 3*df.height.std()
print(lower_limit)

outliers = df[(df.height>upper_limit) | (df.height<lower_limit)]
print(outliers)

final = df[(df.height<upper_limit) & (df.height>lower_limit)]
print(final)

# lets do the same thing with z-score

df['zscore'] = (df.height - df.height.mean())/df.height.std()
print(df)

print(df[df['zscore']>3])
print(df[df['zscore']<-3])

outliers = df[(df.zscore<-3) | (df.zscore>3)]
print(outliers)

final = df[(df.zscore>-3) & (df.zscore<3)]
print(final)
        
