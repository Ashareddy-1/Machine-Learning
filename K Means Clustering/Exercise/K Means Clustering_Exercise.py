#Use iris flower dataset from sklearn library and try to form clusters of flowers using petal width and length features. Drop other two features for simplicity.
#Figure out if any preprocessing such as scaling would help here
#Draw elbow plot and from that figure out optimal value of k

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)

required = df.drop(['sepal length (cm)', 'sepal width (cm)'], axis='columns')
print(required)


plt.scatter(required['petal width (cm)'], required['petal length (cm)'])
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Petal Width vs Petal Length')
plt.show()

km = KMeans(n_clusters=3)

y_predicted = km.fit_predict(required[['petal width (cm)', 'petal length (cm)']])
print(y_predicted)

required['cluster'] = y_predicted
print(required)

df1 = required[required.cluster==0]
df2 = required[required.cluster==1]
df3 = required[required.cluster==2]

plt.scatter(df1['petal width (cm)'], df1['petal length (cm)'],color='green')
plt.scatter(df2['petal width (cm)'], df2['petal length (cm)'],color='red')
plt.scatter(df3['petal width (cm)'], df3['petal length (cm)'],color='black')

plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Petal Width vs Petal Length')
plt.show()

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple', marker = '+', label = 'centroid')

plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Petal Width vs Petal Length')
plt.show()

k_rng = range(1,20)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal width (cm)', 'petal length (cm)']])
    sse.append(km.inertia_)
    print(sse)

plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.plot(k_rng,sse)
plt.show()

