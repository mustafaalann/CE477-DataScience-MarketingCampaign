import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as per
from sklearn.cluster import DBSCAN


#loading data
dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['Income','MntWines']].dropna()
print('Before Standardization')
print(columnsToRescale)

#Standardization
scaler = per.StandardScaler().fit(columnsToRescale)
standardizedData = scaler.transform(columnsToRescale)
standardizedData = pd.DataFrame(standardizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Standardization')
print(standardizedData)
X = standardizedData

clusterer = DBSCAN(eps=0.3, min_samples=10, metric='euclidean')
y_pred = clusterer.fit_predict(X)

plt.figure(figsize=(12,9))
plt.annotate('CE 477', xy=(0.03, 0.95), xycoords='axes fraction')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='Dark2')