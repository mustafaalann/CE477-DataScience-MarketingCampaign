import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#loading dataset
from sklearn.preprocessing import Normalizer

dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['Income','MntWines']].dropna()
print('Before Normalization')
print(columnsToRescale)

#Normalization of dataset
scaler = Normalizer().fit(columnsToRescale)
normalizedData = scaler.transform(columnsToRescale)
normalizedData = pd.DataFrame(normalizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Normalization')
print(normalizedData)
#initialized kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 100,
"random_state": 1,
}

#created list to hold SSE(sum of the squared Euclidean distances) values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(normalizedData)
    sse.append(kmeans.inertia_)

#visualized results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
#instantiated the k-means via using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=3, n_init=100, random_state=1)

#fit k-means algorithm to dataset
kmeans.fit(normalizedData)

#checked cluster for observation
kmeans.labels_


#appended cluster assingments to normalizedData
normalizedData['cluster'] = kmeans.labels_

#viewed updated DataSet
print(normalizedData)
#viewed Centroids
centroids = kmeans.cluster_centers_
print(centroids)